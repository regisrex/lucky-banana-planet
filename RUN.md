# kinya — RBA radio → Kinyarwanda ASR test harness

Streams live Rwanda Broadcasting Agency radio into a speech-to-text backend and
logs transcripts with latency, so Pindo / local open models can be compared on
identical broadcast audio.

## Quick start

```bash
brew install ffmpeg
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./make_ca_bundle.sh                 # REQUIRED for the https stations — see TLS below

.venv/bin/python rba_stt.py --list
.venv/bin/python rba_stt.py --station rwanda --backend dump  --minutes 1
.venv/bin/python rba_stt.py --station rwanda --backend local --minutes 5
PINDO_TOKEN=... .venv/bin/python rba_stt.py --station inteko --backend pindo --minutes 5
```

Output is JSONL in `runs/`, one record per chunk, with `latency_s`, `rtf`
(latency ÷ chunk length; <1.0 keeps up with realtime) and `level_dbfs`.

## Stations — verified 2026-10-05

Verified by reading the `icy-name` header off each port and decoding 4s with
ffmpeg. All 8 were live.

| id | station | stream | kbps | notes |
|---|---|---|---|---|
| `rwanda` | Radio Rwanda | `https://listen.rba.co.rw:8008/rwanda` | 128 | national; cleanest anchor speech — best WER baseline |
| `magic` | Magic FM 90.7 | `http://listen.rba.co.rw:8080/` | 160 | **http only**; heavy rw/en/fr code-switching + music |
| `rusizi` | Radio Rusizi | `https://listen.rba.co.rw:8009/rusizi` | 128 | regional, west |
| `huye` | Radio Huye | `https://listen.rba.co.rw:5053/huye` | 160 | regional, south |
| `rubavu` | Radio Rubavu | `https://listen.rba.co.rw:8004/rubavu` | 96 | low-bitrate stress test |
| `inteko` | Radio Inteko 101.5 | `https://listen.rba.co.rw:8007/inteko` | 48 | parliament; worst bitrate, but formal speech & no music |
| `musanze` | Radio Musanze | `https://listen.rba.co.rw:8003/musanze` | 128 | name unconfirmed — see below |
| `nyagatare` | Radio Nyagatare | `https://listen.rba.co.rw:3053/nyagatare` | 128 | name unconfirmed — see below |

### Gotchas worth knowing

**Stations are keyed by PORT, not path.** The server is Shoutcast DNAS 2.6.0, not
Icecast. Any path on a live port returns that station's audio — `:8008/status-json.xsl`
returns MP3, and so does `:8008/does-not-exist`. So an HTTP 200 proves nothing;
`icy-name` is the only real identity check. The paths above are kept only because
that is what RBA's own player pages use.

**`musanze` and `nyagatare` report `icy-name: No Name`.** Their identity comes from
RBA's player pages, not from stream metadata. Treat as probable, not verified.

**Magic FM is http-only.** Port 8080 serves no TLS at all.

### TLS: the server sends an incomplete chain

`listen.rba.co.rw` serves **only its leaf certificate** and omits the RapidSSL
intermediate. curl on macOS appears to work because the system trust store fetches
the missing issuer via AIA; ffmpeg's OpenSSL does not, and fails with:

```
[tls] error:0A000086:SSL routines::certificate verify failed
```

`make_ca_bundle.sh` fetches that intermediate from the cert's own AIA URL and
appends it to the public roots, so verification stays **on**. `rba_stt.py` passes
the result to ffmpeg via `-ca_file` for any `https://` station.

> The leaf cert expires **2026-10-30**. When it is renewed, re-run
> `./make_ca_bundle.sh`; if RBA fixes their chain, the bundle becomes unnecessary.

## Backends

| `--backend` | what it does |
|---|---|
| `pindo` | POSTs each chunk to `https://api.pindo.io/ai/stt/<lang>`. Needs `PINDO_TOKEN`. |
| `local` | transformers ASR pipeline, offline. Default `badrex/w2v-bert-2.0-kinyarwanda-asr`. |
| `dump` | writes chunks to disk only — use to build a fixed eval set both backends can share. |

Pindo's endpoint and `Bearer` auth are verified live (it returns
`401 Bearer token required`). The **multipart field name is a guess** (`audio`) —
there is no public SDK to confirm it, so override with `--field` if you get a 400.
Note the 401 fires *before* language validation, so a 401 on `/sw` or `/lg` does
not prove those languages exist.

## Kinyarwanda ASR options

| model / service | type | notes |
|---|---|---|
| `badrex/w2v-bert-2.0-kinyarwanda-asr` | open, 2.3 GB | **most used by far** (42.7k downloads); CC-BY-4.0; fine-tuned from `facebook/w2v-bert-2.0` on 1000h Kinyarwanda |
| `mbazaNLP/Whisper-Small-Kinyarwanda` | open | Digital Umuganda; 5.8k downloads |
| `DigitalUmuganda/whisper_small_kinyarwanda` | open | smaller DU release |
| `facebook/mms-1b-all` | open | wav2vec2, supports `kin` |
| Pindo VoiceAI | API | native `rw`; file-based REST, not websocket |
| Google Cloud STT | API | `rw-RW` supported; has true streaming gRPC |
| Proto.cx | commercial | claims Kinyarwanda engine; unverified independently |

**Azure has no Kinyarwanda** — checked their STT language list, no `rw-RW`.

## Why chunked, not streaming

Pindo's STT is a file-based REST endpoint, so continuous audio has to be sliced.
The harness therefore cuts fixed chunks (`--chunk`, default 8s) and sends them
through a small worker pool (`--workers`). A crude level gate (`--gate-dbfs`)
drops dead air before it costs an API call; it is **not** a real VAD and will not
reject music. If music-driven hallucination becomes a problem, put a proper VAD
(e.g. Silero) in front of the queue.

## Measured: local backend on Apple Silicon (2026-10-05)

`badrex/w2v-bert-2.0-kinyarwanda-asr` on MPS, 6s chunks captured live from Radio Rwanda:

| | |
|---|---|
| one-time model download + load | ~750 s (2.3 GB) |
| first inference (warmup) | 3.00 s — RTF 0.50 |
| steady-state inference | **0.20 s per 6s chunk — RTF 0.03** |

RTF 0.03 is ~33× faster than realtime, so a single Mac keeps up with a live
stream with enormous headroom — the bottleneck is the network, not inference.
Transcripts were coherent Kinyarwanda news speech off a 128kbps broadcast.

Caveat: w2v-bert is a CTC model with no language model attached, so output is
lowercase and unpunctuated. For punctuation/casing, use a Whisper-based
Kinyarwanda checkpoint instead.

## Live dashboard

A local web UI showing the stations, live transcription, and this Mac's vitals —
built to be demoed over a reverse SSH tunnel.

```bash
.venv/bin/python server.py --port 7311          # binds 127.0.0.1 only
ssh -R 9000:localhost:7311 user@public-host     # share as public-host:9000
```

Panels: per-core CPU bars + memory/swap/load/network/top processes (htop-style,
1 Hz over SSE); all 8 stations with live reachability, latency and `icy-name`
confirmation; a **listen** button per station; and a **transcribe** button that
streams live transcripts into the feed with per-chunk latency and RTF.

**Audio is proxied through this host** (`/stream/<id>`). That is not decoration —
it makes every station same-origin, which solves two things at once: RBA's
incomplete TLS chain, and Magic FM being http-only (mixed content, and so blocked
outright, on an https tunnel).

### Exposure

`server.py` binds `127.0.0.1` by default, so the tunnel decides who gets in and
nothing is open on your LAN. `--host 0.0.0.0` overrides that deliberately.

Be aware of what the page publishes: hostname, CPU model, memory/swap pressure,
load, network throughput, and **your top processes by name**. That is a real
information leak to an untrusted audience. There is no auth — anyone with the
tunnel URL can start and stop transcription. Put it behind the tunnel's own auth,
or a reverse proxy with basic auth, before sharing it widely.

## Switching models: local or Gemini via OpenRouter

The dashboard has a **model** dropdown. Two groups:

- **On this Mac** — the local transformers models (`badrex/w2v-bert`, Whisper-Small).
  Loaded once and cached, so switching between them is instant after the first.
- **OpenRouter (Gemini)** — Gemini 2.5 Flash / Flash-Lite / Pro and 3.8 Flash.
  These are multimodal chat models doing STT as a side effect.

Changing the dropdown while a station is playing swaps the engine live on the same
audio — which is the point: compare w2v-bert against Gemini on the identical stream.

### Enabling the Gemini models

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
.venv/bin/python server.py --port 7311
```

The key is read from the **server's** environment and never sent to the browser.
Without it, the OpenRouter options show greyed out with a hint. The request format
was verified against the live OpenRouter endpoint (`input_audio` base64 WAV → a
401, not a 400, on a dummy key — shape accepted, only auth missing).

### How it differs from local — read before demoing on Gemini

- **Not streaming.** Each chunk is one multimodal chat call, same chunked shape as
  Pindo. OpenRouter chunks default to 8s (vs 5s local) to cut call volume.
- **Latency is network + model**, so expect RTF well above the local 0.04 —
  typically a second or more per chunk. Usable, not instant.
- **It costs money per chunk.** Gemini bills audio input tokens on every call. A
  station left running transcribes continuously and bills continuously. The level
  gate skips silent chunks, but music still costs. Stop it when not demoing.
- **Different failure mode.** w2v-bert is Kinyarwanda-fine-tuned and never refuses;
  Gemini is general multilingual — usually better punctuation and proper nouns,
  but it can occasionally add stray commentary or mishandle heavy music overlap.

New models are just entries in `LOCAL_MODELS` / `OPENROUTER_MODELS` in `server.py`
— any OpenRouter model whose `input_modalities` include `audio` will work.
