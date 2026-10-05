#!/usr/bin/env python3
"""
Live dashboard: RBA radio stations + Kinyarwanda ASR + this Mac's vitals.

    .venv/bin/python server.py --port 7311
    ssh -R 9000:localhost:7311 user@public-host     # then share host:9000

Binds 127.0.0.1 by default so nothing is exposed until you open a tunnel
yourself. --host 0.0.0.0 opts out of that, deliberately.

Endpoints
    /                 dashboard
    /events           SSE: vitals every second + transcripts as they land
    /api/stations     station registry + last known reachability
    /api/asr/start    ?station=rwanda   begin transcribing
    /api/asr/stop     stop
    /stream/<id>      audio proxy - also fixes RBA's broken TLS chain for browsers
"""

import argparse, json, os, queue, socket, ssl, subprocess, sys, threading, time, wave, io
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).parent
CA_BUNDLE = ROOT / "rba-ca-bundle.pem"
SR, CH, WIDTH = 16000, 1, 2
STATIONS = {s["id"]: s for s in json.loads((ROOT / "stations.json").read_text())["stations"]}

try:
    import psutil
except ImportError:
    sys.exit("pip install psutil")

BOOT = psutil.boot_time()
STARTED = time.time()


# ------------------------------------------------------------------ vitals

def _top_procs(n=7):
    out = []
    for p in psutil.process_iter(["name", "cpu_percent", "memory_info"]):
        try:
            cpu = p.info["cpu_percent"] or 0.0
            if cpu <= 0:
                continue
            out.append({
                "name": (p.info["name"] or "?")[:22],
                "cpu": round(cpu, 1),
                "mem_mb": round((p.info["memory_info"].rss if p.info["memory_info"] else 0) / 1e6),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(out, key=lambda x: -x["cpu"])[:n]


_net_last = {"t": time.time(), "rx": 0, "tx": 0}


def vitals():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    n = psutil.net_io_counters()
    now = time.time()
    dt = max(now - _net_last["t"], 1e-6)
    rx = (n.bytes_recv - _net_last["rx"]) / dt if _net_last["rx"] else 0
    tx = (n.bytes_sent - _net_last["tx"]) / dt if _net_last["tx"] else 0
    _net_last.update(t=now, rx=n.bytes_recv, tx=n.bytes_sent)
    try:
        load = os.getloadavg()
    except OSError:
        load = (0, 0, 0)
    return {
        "cores": [round(c, 1) for c in psutil.cpu_percent(percpu=True)],
        "cpu": round(psutil.cpu_percent(), 1),
        "mem": {"used": vm.used, "total": vm.total, "pct": vm.percent},
        "swap": {"used": sw.used, "total": sw.total, "pct": sw.percent},
        "load": [round(x, 2) for x in load],
        "net": {"rx": int(rx), "tx": int(tx)},
        "uptime": int(now - BOOT),
        "procs": _top_procs(),
        "ts": now,
    }


# ------------------------------------------------------------------ ASR

# ------------------------------------------------------------------ models

import base64

LOCAL_MODELS = [
    {"id": "local:badrex/w2v-bert-2.0-kinyarwanda-asr",
     "label": "w2v-bert Kinyarwanda", "note": "local · fine-tuned · fast · no punctuation"},
    {"id": "local:mbazaNLP/Whisper-Small-Kinyarwanda",
     "label": "Whisper-Small Kinyarwanda", "note": "local · punctuation · slower first load"},
]

# Gemini models on OpenRouter that accept audio input (verified against the live
# /models endpoint). All are multimodal chat models doing STT as a side effect,
# so they need OPENROUTER_API_KEY and they bill per call.
OPENROUTER_MODELS = [
    {"id": "openrouter:google/gemini-2.5-flash",
     "label": "Gemini 2.5 Flash", "note": "openrouter · fast · good default"},
    {"id": "openrouter:google/gemini-2.5-flash-lite",
     "label": "Gemini 2.5 Flash-Lite", "note": "openrouter · cheapest"},
    {"id": "openrouter:google/gemini-2.5-pro",
     "label": "Gemini 2.5 Pro", "note": "openrouter · best quality · pricier"},
    {"id": "openrouter:google/gemini-3.8-flash",
     "label": "Gemini 3.8 Flash", "note": "openrouter · newest flash"},
]
DEFAULT_MODEL = LOCAL_MODELS[0]["id"]

PIPES = {}   # hf model id -> transformers pipeline, cached across station switches

OR_PROMPT = (
    "You are a speech-to-text engine for Rwandan radio. Transcribe the audio "
    "verbatim in Kinyarwanda (keep any English or French words as spoken). "
    "Output ONLY the transcript text, with no quotes, labels, or commentary. "
    "If the audio is music, jingles, or silence with no speech, output nothing."
)


def transcribe_openrouter(model_id, pcm_f32):
    """One chunk -> one multimodal chat call. Audio goes as base64 WAV, the
    OpenAI-style `input_audio` content part OpenRouter forwards to Gemini."""
    import numpy as np
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set on the server")
    pcm16 = np.clip(pcm_f32 * 32768.0, -32768, 32767).astype("<i2").tobytes()
    b64 = base64.b64encode(_wav_bytes(pcm16)).decode()
    body = {
        "model": model_id,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": OR_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe this Kinyarwanda radio clip."},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ]},
        ],
    }
    r = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "kinya radio ASR"},
        method="POST")
    with urllib.request.urlopen(r, timeout=70) as resp:
        j = json.loads(resp.read())
    return (j["choices"][0]["message"]["content"] or "").strip()


def _wav_bytes(pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(CH); w.setsampwidth(WIDTH); w.setframerate(SR)
        w.writeframes(pcm)
    return b.getvalue()


class ASR:
    """Owns the model, the ffmpeg reader, and the transcript ring buffer."""

    def __init__(self, broadcast):
        self.broadcast = broadcast
        self.state = "idle"          # idle | loading | running | error
        self.station = None
        self.model = DEFAULT_MODEL
        self.kind = "local"          # local | openrouter
        self._local_mid = None
        self.detail = ""
        self.history = deque(maxlen=120)
        self.stats = {"chunks": 0, "rtf": None, "last_latency": None}
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def status(self):
        return {"state": self.state, "station": self.station, "model": self.model,
                "kind": self.kind, "detail": self.detail, **self.stats}

    def _prepare(self, model):
        """Make the chosen backend ready. OpenRouter is instant (just needs a key);
        a local model loads once and is cached in PIPES for later switches."""
        self.model = model
        if model.startswith("openrouter:"):
            if not os.environ.get("OPENROUTER_API_KEY", "").strip():
                raise RuntimeError("OPENROUTER_API_KEY not set on the server")
            self.kind = "openrouter"
            self.detail = model.split(":", 1)[1] + " · via OpenRouter"
            return
        self.kind = "local"
        mid = model.split(":", 1)[1] if model.startswith("local:") else model
        self._local_mid = mid
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        if mid not in PIPES:
            self.state, self.detail = "loading", f"loading {mid}"
            self.broadcast("asr", self.status())
            from transformers import pipeline
            PIPES[mid] = pipeline("automatic-speech-recognition", model=mid, device=dev)
        self.detail = f"{mid.split('/')[-1]} on {dev}"

    def _transcribe(self, a):
        if self.kind == "openrouter":
            return transcribe_openrouter(self.model.split(":", 1)[1], a)
        out = PIPES[self._local_mid]({"raw": a, "sampling_rate": SR})
        return ((out or {}).get("text") or "").strip()

    def start(self, sid, model, chunk=None, gate=-45.0):
        """Switching stations keeps the loaded model, so only the first click is slow.
        Changing the model (or station) restarts the reader thread."""
        if chunk is None:
            chunk = 8.0 if model.startswith("openrouter:") else 5.0
        with self._lock:
            if self.state == "loading":
                return False, "model is still loading"
            if self.state == "running" and self.station == sid and self.model == model:
                return True, "already running"
            prev = self._thread if self.state == "running" else None
            if prev:
                self._stop.set()
        if prev:
            prev.join(timeout=12)
        with self._lock:
            self._stop.clear()
            self.station = sid
            self._thread = threading.Thread(
                target=self._run, args=(sid, model, chunk, gate), daemon=True)
            self._thread.start()
            return True, "starting"

    def stop(self):
        self._stop.set()
        self.state = "idle"
        self.station = None
        self.broadcast("asr", self.status())

    def _run(self, sid, model, chunk, gate):
        import numpy as np
        st = STATIONS[sid]
        try:
            self._prepare(model)
        except Exception as e:
            self.state, self.detail = "error", f"{type(e).__name__}: {e}"
            self.broadcast("asr", self.status()); return

        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_at_eof", "1",
               "-reconnect_delay_max", "10"]
        if st["url"].startswith("https://"):
            cmd += ["-ca_file", str(CA_BUNDLE)]
        cmd += ["-i", st["url"], "-vn", "-f", "s16le", "-ar", str(SR),
                "-ac", str(CH), "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                bufsize=10 ** 7)
        self.state = "running"
        self.stats = {"chunks": 0, "rtf": None, "last_latency": None}
        self.broadcast("asr", self.status())

        need = int(chunk * SR * CH * WIDTH)
        try:
            while not self._stop.is_set():
                buf = bytearray()
                while len(buf) < need and not self._stop.is_set():
                    part = proc.stdout.read(need - len(buf))
                    if not part:
                        break
                    buf.extend(part)
                if len(buf) < need:
                    break
                a = np.frombuffer(bytes(buf), "<i2").astype("float32") / 32768.0
                rms = float(np.sqrt(np.mean(a * a)) + 1e-9)
                db = 20 * np.log10(rms)
                if db < gate:
                    continue
                t0 = time.time()
                try:
                    text = self._transcribe(a)
                except Exception as e:
                    self.state = "error"
                    self.detail = f"{type(e).__name__}: {str(e)[:120]}"
                    self.broadcast("asr", self.status())
                    break
                lat = time.time() - t0
                if not text:
                    continue
                self.stats["chunks"] += 1
                self.stats["last_latency"] = round(lat, 2)
                self.stats["rtf"] = round(lat / chunk, 3)
                rec = {"t": time.strftime("%H:%M:%S"), "station": sid, "text": text,
                       "latency": round(lat, 2), "rtf": round(lat / chunk, 3),
                       "db": round(float(db), 1)}
                self.history.append(rec)
                self.broadcast("tx", rec)
                self.broadcast("asr", self.status())
        finally:
            proc.terminate()
            if self.state != "error":
                self.state, self.station = "idle", None
            self.broadcast("asr", self.status())


# ------------------------------------------------------------------ SSE hub

class Hub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def add(self):
        q = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.append(q)
        return q

    def drop(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def send(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        with self.lock:
            dead = []
            for q in self.clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.clients.remove(q)


HUB = Hub()
ASRS = ASR(HUB.send)

# station reachability, refreshed in the background
REACH = {k: {"up": None, "checked": 0} for k in STATIONS}


def _ssl_ctx():
    ctx = ssl.create_default_context()
    if CA_BUNDLE.exists():
        ctx.load_verify_locations(str(CA_BUNDLE))
    return ctx


def probe(sid):
    st = STATIONS[sid]
    req = urllib.request.Request(st["url"], headers={"User-Agent": "kinya-dash", "Icy-MetaData": "0"})
    try:
        t0 = time.time()
        ctx = _ssl_ctx() if st["url"].startswith("https") else None
        with urllib.request.urlopen(req, timeout=8, context=ctx) as r:
            r.read(2048)
            return {"up": True, "ms": int((time.time() - t0) * 1000),
                    "icy": r.headers.get("icy-name"), "checked": time.time()}
    except Exception as e:
        return {"up": False, "err": f"{type(e).__name__}", "checked": time.time()}


def reach_loop():
    while True:
        for sid in STATIONS:
            REACH[sid] = probe(sid)
            HUB.send("station", {"id": sid, **REACH[sid]})
        time.sleep(45)


def vitals_loop():
    psutil.cpu_percent(percpu=True)
    while True:
        HUB.send("vitals", vitals())
        time.sleep(1)


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kinya"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p == "/":
            f = ROOT / "static" / "index.html"
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")

        if p == "/api/stations":
            return self._send(200, {
                "stations": [{**STATIONS[k], **REACH.get(k, {})} for k in STATIONS],
                "asr": ASRS.status(),
                "host": {"name": socket.gethostname(),
                         "cpu": _cpu_brand(), "cores": psutil.cpu_count()},
            })

        if p == "/api/models":
            return self._send(200, {
                "local": LOCAL_MODELS,
                "openrouter": OPENROUTER_MODELS,
                "default": DEFAULT_MODEL,
                "openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
            })

        if p == "/api/transcripts":
            return self._send(200, {"items": list(ASRS.history)})

        if p == "/api/asr/start":
            sid = (q.get("station") or ["rwanda"])[0]
            if sid not in STATIONS:
                return self._send(400, {"error": "unknown station"})
            model = (q.get("model") or [DEFAULT_MODEL])[0]
            ok, msg = ASRS.start(sid, model)
            return self._send(200 if ok else 409, {"ok": ok, "msg": msg})

        if p == "/api/asr/stop":
            ASRS.stop()
            return self._send(200, {"ok": True})

        if p == "/events":
            return self._sse()

        if p.startswith("/stream/"):
            return self._proxy(p.rsplit("/", 1)[-1])

        return self._send(404, {"error": "not found"})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        qq = HUB.add()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.write(f"event: vitals\ndata: {json.dumps(vitals())}\n\n".encode())
            self.wfile.write(f"event: asr\ndata: {json.dumps(ASRS.status())}\n\n".encode())
            for r in list(ASRS.history)[-25:]:
                self.wfile.write(f"event: tx\ndata: {json.dumps(r)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    self.wfile.write(qq.get(timeout=20))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.drop(qq)

    def _proxy(self, sid):
        """Relay the Shoutcast MP3 to the browser.

        Two problems this solves: RBA omits its TLS intermediate (browsers behind
        a tunnel may balk), and Magic FM is http-only, which is mixed content on
        an https tunnel. Proxying makes every station same-origin.
        """
        if sid not in STATIONS:
            return self._send(404, {"error": "unknown station"})
        url = STATIONS[sid]["url"]
        req = urllib.request.Request(url, headers={"User-Agent": "kinya-dash", "Icy-MetaData": "0"})
        try:
            ctx = _ssl_ctx() if url.startswith("https") else None
            up = urllib.request.urlopen(req, timeout=12, context=ctx)
        except Exception as e:
            return self._send(502, {"error": f"upstream: {type(e).__name__}"})
        self.send_response(200)
        self.send_header("Content-Type", up.headers.get("content-type", "audio/mpeg"))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                b = up.read(8192)
                if not b:
                    break
                self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            up.close()


def _cpu_brand():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7311)
    ap.add_argument("--host", default="127.0.0.1",
                    help="default binds loopback only; the tunnel decides exposure")
    a = ap.parse_args()

    threading.Thread(target=vitals_loop, daemon=True).start()
    threading.Thread(target=reach_loop, daemon=True).start()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"  kinya dashboard  http://{a.host}:{a.port}")
    print(f"  share it:        ssh -R 9000:localhost:{a.port} user@public-host")
    if a.host != "127.0.0.1":
        print("  !! bound beyond loopback - this exposes host vitals on your network")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
