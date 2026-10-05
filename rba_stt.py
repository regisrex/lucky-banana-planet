#!/usr/bin/env python3
"""
Stream RBA radio into a speech-to-text backend and log transcripts + latency.

  python3 rba_stt.py --station rwanda --backend pindo --minutes 5
  python3 rba_stt.py --station inteko --backend local --model badrex/w2v-bert-2.0-kinyarwanda-asr
  python3 rba_stt.py --station magic  --backend dump      # just save wavs, no API calls

Backends:
  pindo  POST each chunk to https://api.pindo.io/ai/stt/<lang>  (needs PINDO_TOKEN)
  local  transformers ASR pipeline, runs offline
  dump   write chunks to disk only - use to build a fixed eval set

Why chunked: Pindo's STT is a file-based REST endpoint, not a websocket, so a
continuous stream has to be sliced. Everything downstream assumes that shape.
"""

import utils.http_session
import utils.transport
import utils.session
import argparse, io, json, os, queue, subprocess, sys, threading, time, wave
from pathlib import Path

ROOT = Path(__file__).parent
SR, CH, WIDTH = 16000, 1, 2  # 16kHz mono s16le - what every ASR backend here wants


def load_stations():
    return {s["id"]: s for s in json.loads((ROOT / "stations.json").read_text())["stations"]}


CA_BUNDLE = ROOT / "rba-ca-bundle.pem"


def ffmpeg_proc(url, verbose=False):
    """Decode the stream to raw PCM on stdout. Reconnect flags keep long runs alive.

    -ca_file is not optional for the https ports: listen.rba.co.rw serves ONLY the
    leaf cert and omits the RapidSSL intermediate. macOS curl hides this by fetching
    the issuer via AIA; ffmpeg's OpenSSL does not, so it fails with
    'certificate verify failed'. make_ca_bundle.sh rebuilds the bundle.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_at_eof", "1",
        "-reconnect_delay_max", "10", "-rw_timeout", "15000000",
        "-user_agent", "Mozilla/5.0 (rba-stt-benchmark)",
    ]
    if url.startswith("https://"):
        if not CA_BUNDLE.exists():
            sys.exit(f"missing {CA_BUNDLE.name} - run ./make_ca_bundle.sh first "
                     "(RBA serves an incomplete cert chain; see README)")
        cmd += ["-ca_file", str(CA_BUNDLE)]
    cmd += [
        "-i", url,
        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(SR), "-ac", str(CH), "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE,
        stderr=None if verbose else subprocess.DEVNULL,
        bufsize=10 ** 7,
    )


def read_exact(stream, n):
    """stdout.read() can short-read on a pipe; loop until the chunk is full."""
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            return bytes(buf) if buf else None
        buf.extend(part)
    return bytes(buf)


def to_wav(pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(CH); w.setsampwidth(WIDTH); w.setframerate(SR)
        w.writeframes(pcm)
    return b.getvalue()


def rms_dbfs(pcm):
    """Cheap level gate. Not a real VAD - it rejects silence/dead air, not music."""
    import array, math
    a = array.array("h"); a.frombytes(pcm)
    if not len(a):
        return -99.0
    mean_sq = sum(float(x) * x for x in a[::17]) / len(a[::17])  # stride = speed
    return 20 * math.log10(math.sqrt(mean_sq) / 32768 + 1e-9)


# ---------------------------------------------------------------- backends

class PindoBackend:
    """Pindo VoiceAI STT. Endpoint + auth scheme verified live (401 'Bearer token
    required'); the multipart FIELD NAME is a guess - override with --field if it 400s."""

    def __init__(self, lang, field="audio", timeout=60):
        import requests
        self.requests = requests
        self.token = os.environ.get("PINDO_TOKEN", "").strip()
        if not self.token:
            sys.exit("PINDO_TOKEN is not set. export PINDO_TOKEN=... first.")
        self.url = f"https://api.pindo.io/ai/stt/{lang}"
        self.field, self.timeout = field, timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def __call__(self, pcm, idx):
        files = {self.field: (f"chunk_{idx:05d}.wav", to_wav(pcm), "audio/wav")}
        r = self.session.post(self.url, files=files, timeout=self.timeout)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}", "body": r.text[:400]}
        try:
            j = r.json()
        except ValueError:
            return {"error": "non-JSON response", "body": r.text[:400]}
        # Shape is undocumented publicly; dig for the text and keep the raw payload.
        text = ""
        for path in (("data", "text"), ("text",), ("data", "transcript"), ("transcript",), ("result",)):
            cur = j
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, str) and cur:
                text = cur
                break
        return {"text": text, "raw": j}


class LocalBackend:
    """transformers ASR pipeline - works for both whisper and w2v-bert checkpoints."""

    def __init__(self, model):
        import numpy as np
        from transformers import pipeline
        self.np = np
        print(f"loading {model} (first run downloads weights)...", file=sys.stderr)
        self.pipe = pipeline("automatic-speech-recognition", model=model)

    def __call__(self, pcm, idx):
        audio = self.np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        out = self.pipe({"raw": audio, "sampling_rate": SR})
        return {"text": (out or {}).get("text", "").strip(), "raw": out}


class DumpBackend:
    def __init__(self, outdir):
        self.dir = Path(outdir); self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, pcm, idx):
        p = self.dir / f"chunk_{idx:05d}.wav"
        p.write_bytes(to_wav(pcm))
        return {"text": "", "saved": str(p)}


# ---------------------------------------------------------------- runner

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="rwanda")
    ap.add_argument("--backend", default="pindo", choices=["pindo", "local", "dump"])
    ap.add_argument("--lang", default=None, help="Pindo language code (default: station's)")
    ap.add_argument("--model", default="badrex/w2v-bert-2.0-kinyarwanda-asr", help="local backend model")
    ap.add_argument("--chunk", type=float, default=8.0, help="chunk seconds")
    ap.add_argument("--minutes", type=float, default=5.0, help="0 = run until Ctrl-C")
    ap.add_argument("--gate-dbfs", type=float, default=-45.0, help="skip chunks quieter than this")
    ap.add_argument("--field", default="audio", help="Pindo multipart field name")
    ap.add_argument("--workers", type=int, default=2, help="parallel in-flight requests")
    ap.add_argument("--out", default=None, help="JSONL output path")
    ap.add_argument("--save-audio", action="store_true", help="also keep each chunk's wav")
    ap.add_argument("--list", action="store_true", help="list stations and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="show ffmpeg stderr")
    args = ap.parse_args()

    stations = load_stations()
    if args.list:
        for s in stations.values():
            print(f"{s['id']:<10} {s['name']:<22} {s['bitrate']:>3}kbps  {s['url']}")
        return
    if args.station not in stations:
        sys.exit(f"unknown station '{args.station}'. options: {', '.join(stations)}")

    st = stations[args.station]
    lang = args.lang or st.get("lang", "rw")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out or ROOT / "runs" / f"{args.station}-{args.backend}-{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = out_path.parent / f"{args.station}-{run_id}-audio"

    if args.backend == "pindo":
        backend = PindoBackend(lang, field=args.field)
    elif args.backend == "local":
        backend = LocalBackend(args.model)
    else:
        backend = DumpBackend(audio_dir)

    chunk_bytes = int(args.chunk * SR * CH * WIDTH)
    deadline = time.time() + args.minutes * 60 if args.minutes else None

    print(f"station : {st['name']}  ({st['bitrate']}kbps, {st['sample_rate']}Hz)")
    print(f"url     : {st['url']}")
    print(f"backend : {args.backend}" + (f" [{lang}]" if args.backend == "pindo" else ""))
    print(f"chunk   : {args.chunk}s   gate: {args.gate_dbfs} dBFS   out: {out_path}")
    print("-" * 72)

    q, stats = queue.Queue(maxsize=args.workers * 4), {"sent": 0, "ok": 0, "err": 0, "skip": 0, "lat": []}
    lock = threading.Lock()
    fh = out_path.open("a", buffering=1)

    def worker():
        while True:
            item = q.get()
            if item is None:
                q.task_done(); return
            idx, pcm, captured_at, level = item
            t0 = time.time()
            try:
                res = backend(pcm, idx)
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {e}"}
            lat = time.time() - t0
            rec = {
                "chunk": idx, "station": args.station, "backend": args.backend,
                "captured_at": captured_at, "latency_s": round(lat, 3),
                "rtf": round(lat / args.chunk, 3), "level_dbfs": round(level, 1),
                **res,
            }
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if "error" in rec:
                    stats["err"] += 1
                    print(f"[{idx:04d}] ERROR {rec['error']} {str(rec.get('body',''))[:120]}")
                else:
                    stats["ok"] += 1; stats["lat"].append(lat)
                    txt = rec.get("text") or "(empty)"
                    print(f"[{idx:04d}] {lat:5.2f}s {level:6.1f}dB  {txt}")
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    proc = ffmpeg_proc(st["url"], verbose=args.verbose)
    idx = 0
    try:
        while True:
            if deadline and time.time() > deadline:
                print("\n-- time limit reached --"); break
            pcm = read_exact(proc.stdout, chunk_bytes)
            if pcm is None or len(pcm) < chunk_bytes:
                print("\n-- stream ended; ffmpeg will have retried already --"); break
            idx += 1
            level = rms_dbfs(pcm)
            if level < args.gate_dbfs:
                stats["skip"] += 1
                print(f"[{idx:04d}] skip (silent, {level:.1f} dBFS)")
                continue
            if args.save_audio and args.backend != "dump":
                audio_dir.mkdir(parents=True, exist_ok=True)
                (audio_dir / f"chunk_{idx:05d}.wav").write_bytes(to_wav(pcm))
            stats["sent"] += 1
            q.put((idx, pcm, time.strftime("%Y-%m-%dT%H:%M:%S"), level))
    except KeyboardInterrupt:
        print("\n-- interrupted --")
    finally:
        proc.terminate()
        q.join()
        for _ in threads:
            q.put(None)
        fh.close()
        lat = stats["lat"]
        print("-" * 72)
        print(f"chunks sent {stats['sent']}  ok {stats['ok']}  errors {stats['err']}  skipped {stats['skip']}")
        if lat:
            lat_s = sorted(lat)
            p50 = lat_s[len(lat_s) // 2]
            p95 = lat_s[min(len(lat_s) - 1, int(len(lat_s) * 0.95))]
            print(f"latency p50 {p50:.2f}s  p95 {p95:.2f}s  mean RTF {sum(lat)/len(lat)/args.chunk:.2f}")
            print(f"  (RTF < 1.0 means it keeps up with realtime)")
        print(f"transcripts -> {out_path}")


if __name__ == "__main__":
    main()
