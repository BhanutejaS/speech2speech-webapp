import os
import csv
import queue
import threading
import time
import pathlib
import difflib
from statistics import mean
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from kokoro import KPipeline

from flask import Flask, render_template, jsonify

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

# ----------------------------- CONFIG -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000
BLOCK_SIZE = 512

# VAD / segmentation
RMS_HISTORY = 30
SILENCE_MULTIPLIER = 1.6
MIN_CHUNK_SEC = 0.80
PAUSE_TO_CUT_SEC = 0.90

# Whisper
WHISPER_MODEL = "tiny.en"
WHISPER_ARGS = dict(
    fp16=False,
    language="en",
    condition_on_previous_text=False,
    temperature=0.0,
    no_speech_threshold=0.6,
    logprob_threshold=-0.5,
)

# LLM
LLM_MODEL = "gpt-4o-mini"

# Kokoro TTS
VOICE = "af_heart"
OUT_DIR = pathlib.Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pipe = KPipeline(lang_code="a")

# Benchmarks file
CSV_LOG_PATH = Path("latency_log.csv")

# -------------------------- STATE / QUEUES ------------------------
audio_q = queue.Queue()
# tts_q carries dicts: {"turn_id": int, "audio": np.ndarray}
tts_q = queue.Queue()
web_q = queue.Queue()
stop_event = threading.Event()
listening_event = threading.Event()  # controls mic listening (pause while AI talks)

rms_values = []
conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
last_user_text = ""
LLM_LOCK = threading.Lock()

# --------------------- LATENCY / BENCHMARK STATE ------------------
TURN_COUNTER = 0
# turn_id -> dict of timings & metadata
LAT = {}
LAT_LOCK = threading.Lock()
# keys we record per turn:
# "e2e_start","stt_time","llm_start","llm_ttft","llm_total",
# "tts_ttft_play","e2e_ttft","user_text","assistant_text","_logged"

def _new_turn():
    global TURN_COUNTER
    with LAT_LOCK:
        TURN_COUNTER += 1
        tid = TURN_COUNTER
        LAT[tid] = {}
    return tid

def _lat_set(turn_id, key, value):
    with LAT_LOCK:
        if turn_id in LAT:
            LAT[turn_id][key] = value

def _lat_get(turn_id, key, default=None):
    with LAT_LOCK:
        return LAT.get(turn_id, {}).get(key, default)

def _have_all_core_metrics(d):
    keys = ("stt_time", "llm_ttft", "llm_total", "tts_ttft_play", "e2e_ttft")
    return all(isinstance(d.get(k), (int, float)) for k in keys)

def _maybe_log_csv(turn_id):
    """Write a single row to CSV once per turn when we have all metrics."""
    with LAT_LOCK:
        d = LAT.get(turn_id, {})
        if not d or d.get("_logged"):
            return
        if not _have_all_core_metrics(d):
            return

        is_new = not CSV_LOG_PATH.exists()
        with CSV_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow([
                    "turn_id","stt_time","llm_ttft","llm_total",
                    "tts_ttft_play","e2e_ttft","user_text_len","assistant_len"
                ])
            w.writerow([
                turn_id,
                d.get("stt_time"),
                d.get("llm_ttft"),
                d.get("llm_total"),
                d.get("tts_ttft_play"),
                d.get("e2e_ttft"),
                len((d.get("user_text") or "")),
                len((d.get("assistant_text") or "")),
            ])
        d["_logged"] = True

def _lat_rows():
    with LAT_LOCK:
        rows = []
        for tid, d in sorted(LAT.items()):
            rows.append({
                "turn_id": tid,
                "stt_time": d.get("stt_time"),
                "llm_ttft": d.get("llm_ttft"),
                "llm_total": d.get("llm_total"),
                "tts_ttft_play": d.get("tts_ttft_play"),
                "e2e_ttft": d.get("e2e_ttft"),
                "user_text": d.get("user_text"),
                "assistant_len": len(d.get("assistant_text", "")) if d.get("assistant_text") else 0,
            })
        return rows

def _lat_summarize():
    rows = _lat_rows()
    def _avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(mean(vals), 4) if vals else None
    summary = {
        "avg_stt_time": _avg("stt_time"),
        "avg_llm_ttft": _avg("llm_ttft"),
        "avg_llm_total": _avg("llm_total"),
        "avg_tts_ttft_play": _avg("tts_ttft_play"),
        "avg_e2e_ttft": _avg("e2e_ttft"),
        "turns_counted": len(rows),
    }
    return rows, summary

def _percentiles(values):
    if not values:
        return None, None, None
    vals = sorted(values)
    def q(p):
        idx = (len(vals)-1)*p
        lo, hi = int(idx), min(int(idx)+1, len(vals)-1)
        w = idx - lo
        return round(vals[lo]*(1-w) + vals[hi]*w, 4)
    return q(0.5), q(0.9), q(0.95)

def _summary_with_percentiles(rows):
    def col(k):
        return [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    summary = {}
    for k in ["stt_time","llm_ttft","llm_total","tts_ttft_play","e2e_ttft"]:
        vals = col(k)
        p50, p90, p95 = _percentiles(vals)
        summary[k] = {
            "avg": round(mean(vals), 4) if vals else None,
            "p50": p50, "p90": p90, "p95": p95, "n": len(vals)
        }
    return summary

def _render_latency_html(rows, summary):
    def fmt(x):
        return "" if x is None else f"{x:.4f}"
    head = """
    <html><head><meta charset="utf-8">
    <title>Latency Report</title>
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;}
      h1{margin:0 0 8px;}
      table{border-collapse:collapse;width:100%;margin-top:12px;}
      th,td{border:1px solid #ddd;padding:8px;font-size:14px}
      th{background:#f3f3f3;text-align:left}
      .mono{font-family:ui-monospace,Consolas,monospace}
      .grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0;}
      .card{border:1px solid #eee;border-radius:8px;padding:12px;background:#fafafa}
      .muted{color:#666}
      a.btn{display:inline-block;margin-top:8px;padding:8px 12px;border:1px solid #ccc;border-radius:6px;text-decoration:none;color:#222;background:#fff}
    </style>
    </head><body>
    """
    top = f"<h1>Latency Report</h1><div class='muted'>Turns counted: {len(rows)}</div>"
    labels = {
        "stt_time":"STT",
        "llm_ttft":"LLM TTFT",
        "llm_total":"LLM Total",
        "tts_ttft_play":"TTS TTFT (play)",
        "e2e_ttft":"End-to-End TTFT"
    }
    cards = []
    for k, lab in labels.items():
        s = summary.get(k, {})
        cards.append(
            f"<div class='card'><div><b>{lab}</b></div>"
            f"<div>Avg: <span class='mono'>{fmt(s.get('avg'))}</span> s</div>"
            f"<div>P50: <span class='mono'>{fmt(s.get('p50'))}</span> s</div>"
            f"<div>P90: <span class='mono'>{fmt(s.get('p90'))}</span> s</div>"
            f"<div>P95: <span class='mono'>{fmt(s.get('p95'))}</span> s</div>"
            f"<div class='muted'>n={s.get('n','')}</div></div>"
        )
    grid = "<div class='grid'>" + "".join(cards) + "</div>"

    th = ("<tr>"
          "<th>Turn</th><th>STT</th><th>LLM TTFT</th><th>LLM Total</th>"
          "<th>TTS TTFT(play)</th><th>E2E TTFT</th><th>User text</th><th>Assistant len</th></tr>")
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td class='mono'>{r.get('turn_id')}</td>"
            f"<td class='mono'>{fmt(r.get('stt_time'))}</td>"
            f"<td class='mono'>{fmt(r.get('llm_ttft'))}</td>"
            f"<td class='mono'>{fmt(r.get('llm_total'))}</td>"
            f"<td class='mono'>{fmt(r.get('tts_ttft_play'))}</td>"
            f"<td class='mono'>{fmt(r.get('e2e_ttft'))}</td>"
            f"<td class='mono'>{(r.get('user_text') or '')[:80]}</td>"
            f"<td class='mono'>{r.get('assistant_len')}</td>"
            "</tr>"
        )
    table = "<table>" + th + "".join(trs) + "</table>"
    links = "<a class='btn' href='/latency_stats'>JSON</a> <a class='btn' href='/latency_stats.csv'>Download CSV</a>"
    return head + top + links + grid + table + "</body></html>"

# -------------------------- AUDIO HELPERS -------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

def is_speech(audio_chunk):
    rms = float(np.sqrt(np.mean(audio_chunk**2) + 1e-10))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

# ---------------------------- TTS --------------------------------
def synth_kokoro_stream(text: str, voice: str = VOICE):
    if not text or not text.strip():
        return
    for _gs, _ps, audio in pipe(text.strip(), voice=voice):
        if audio is None or len(audio) == 0:
            continue
        audio = np.asarray(audio, dtype=np.float32)
        m = np.max(np.abs(audio))
        if m > 0:
            audio = audio / m
        yield audio

def speak_by_clauses(text: str, turn_id: int):
    clauses = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".?!;:" and len(buf.strip()) > 0:
            clauses.append(buf.strip())
            buf = ""
    if buf.strip():
        clauses.append(buf.strip())

    for clause in clauses:
        for chunk in synth_kokoro_stream(clause):
            if stop_event.is_set():
                return
            tts_q.put({"turn_id": turn_id, "audio": chunk})

# ------------------------ TTS WORKER (PLAY) -----------------------
def tts_worker():
    print("[TTS worker] started")
    seen_first_play = set()

    while True:
        try:
            if stop_event.is_set():
                print("\n[TTS Stop Signal Received]")
                sd.stop()
                with tts_q.mutex:
                    tts_q.queue.clear()
                stop_event.clear()
                continue

            item = tts_q.get(timeout=0.1)
            if not isinstance(item, dict) or "audio" not in item:
                tts_q.task_done()
                continue

            tid = item.get("turn_id")
            audio = item["audio"]

            # First chunk actually about to PLAY -> log TTS TTFT & E2E TTFT
            if tid and tid not in seen_first_play:
                seen_first_play.add(tid)
                now = time.time()
                _lat_set(tid, "tts_ttft_play", round(now - _lat_get(tid, "llm_start", now), 4))
                e2e_start = _lat_get(tid, "e2e_start")
                if e2e_start:
                    _lat_set(tid, "e2e_ttft", round(now - e2e_start, 4))
                print(f"[Latency][Turn {tid}] TTS TTFT(play): {_lat_get(tid,'tts_ttft_play')} s | "
                      f"E2E TTFT: {_lat_get(tid,'e2e_ttft')} s")
                _maybe_log_csv(tid)

            sd.play(audio, 24000, blocking=False)
            while sd.get_stream().active:
                if stop_event.is_set():
                    break
                time.sleep(0.01)

            if stop_event.is_set():
                sd.stop()
                with tts_q.mutex:
                    tts_q.queue.clear()
                stop_event.clear()

            tts_q.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS worker error: {e}")

threading.Thread(target=tts_worker, daemon=True).start()

# ----------------------------- LLM --------------------------------
def speculative_gpt_tts_stream(segment_text, turn_id: int):
    global conversation_context
    with LLM_LOCK:
        web_q.put({"role": "user", "content": segment_text})
        conversation_context.append({"role": "user", "content": segment_text})
        print(f"\n[STT -> GPT][Turn {turn_id}] {segment_text}")

        # Pause mic while AI generates/speaks
        listening_event.clear()

        try:
            buf = ""
            llm_start = time.time()
            _lat_set(turn_id, "llm_start", llm_start)

            first_token_time = None
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True
            )
            for chunk in resp:
                if stop_event.is_set():
                    print("\n[LLM Generation Stopped]")
                    break
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                        _lat_set(turn_id, "llm_ttft", round(first_token_time - llm_start, 4))
                        print(f"[Latency][Turn {turn_id}] LLM TTFT: {_lat_get(turn_id,'llm_ttft')} s")
                    buf += delta.content
                    print(delta.content, end="", flush=True)

            llm_end = time.time()
            _lat_set(turn_id, "llm_total", round(llm_end - llm_start, 4))
            print(f"\n[GPT Complete][Turn {turn_id}] LLM total: {_lat_get(turn_id,'llm_total')} s")

            web_q.put({"role": "assistant", "content": buf})
            _lat_set(turn_id, "assistant_text", buf)

            if not stop_event.is_set():
                speak_by_clauses(buf, turn_id)
                conversation_context.append({"role": "assistant", "content": buf})

        except Exception as e:
            print(f"[GPT/TTS Error][Turn {turn_id}] {e}")
        finally:
            tts_q.join()      # ensure TTS finished before resuming mic
            listening_event.set()
            print(f"[Listening Resumed][Turn {turn_id}]")

# --------------------------- CORE THREAD ---------------------------
def live_transcribe_loop():
    global last_user_text
    print("Core thread started...")
    model = whisper.load_model(WHISPER_MODEL)
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        listening_event.set()
        while True:
            listening_event.wait()
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback
            ):
                while listening_event.is_set():
                    audio = audio_q.get()
                    buffer = np.concatenate((buffer, audio))

                    if is_speech(audio):
                        last_speech_time = time.time()

                    enough_audio = buffer.size >= int(MIN_CHUNK_SEC * SAMPLE_RATE)
                    paused_long = last_speech_time and (time.time() - last_speech_time > PAUSE_TO_CUT_SEC)

                    if enough_audio and paused_long:
                        turn_id = _new_turn()
                        e2e_start = time.time()
                        _lat_set(turn_id, "e2e_start", e2e_start)

                        audio_chunk = buffer.copy()
                        buffer = np.zeros((0,), dtype=np.float32)
                        last_speech_time = None

                        # STT timing
                        stt_t0 = time.time()
                        result = model.transcribe(audio_chunk, **WHISPER_ARGS)
                        stt_t1 = time.time()
                        stt_time = round(stt_t1 - stt_t0, 4)
                        _lat_set(turn_id, "stt_time", stt_time)

                        text = (result.get("text") or "").strip()
                        _lat_set(turn_id, "user_text", text)
                        print(f"[Latency][Turn {turn_id}] STT: {stt_time} s | Text: {text!r}")

                        if text:
                            sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                            if sim > 0.85 or text.lower() in last_user_text.lower():
                                continue
                            last_user_text = text

                            stop_event.clear()

                            if not LLM_LOCK.locked():
                                threading.Thread(
                                    target=speculative_gpt_tts_stream,
                                    args=(text, turn_id),
                                    daemon=True
                                ).start()

                            rms_values.clear()

    except KeyboardInterrupt:
        print("\nStopped.")
        tts_q.put(None)
    except Exception as e:
        print(f"Core loop error: {e}")

core_thread = threading.Thread(target=live_transcribe_loop, daemon=True)
core_thread.start()

# --------------------------- FLASK APP ---------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_updates")
def get_updates():
    messages = []
    while not web_q.empty():
        messages.append(web_q.get())
    return jsonify(messages=messages)

@app.route("/stop_speaking")
def stop_speaking():
    print("Received Stop Signal from Web UI")
    stop_event.set()
    listening_event.set()
    return jsonify(success=True)

@app.route("/latency_stats")
def latency_stats():
    rows, summary = _lat_summarize()
    return jsonify(summary=summary, rows=rows)

@app.route("/latency_report")
def latency_report():
    rows = _lat_rows()
    summary = _summary_with_percentiles(rows)
    html = _render_latency_html(rows, summary)
    return html

@app.route("/latency_stats.csv")
def latency_stats_csv():
    if not CSV_LOG_PATH.exists():
        # build CSV from memory if file not written yet
        rows = _lat_rows()
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["turn_id","stt_time","llm_ttft","llm_total","tts_ttft_play","e2e_ttft","user_text_len","assistant_len"])
        for r in rows:
            w.writerow([
                r.get("turn_id"),
                r.get("stt_time"),
                r.get("llm_ttft"),
                r.get("llm_total"),
                r.get("tts_ttft_play"),
                r.get("e2e_ttft"),
                len((r.get("user_text") or "")),
                r.get("assistant_len"),
            ])
        return buf.getvalue(), 200, {"Content-Type":"text/csv; charset=utf-8"}
    return CSV_LOG_PATH.read_text(encoding="utf-8"), 200, {"Content-Type":"text/csv; charset=utf-8"}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

'''
* turn_id :-
 - The conversation turn number (each time you spoke and the system replied = one turn).

* stt_time :-
 - Time (in seconds) Whisper needed to transcribe your speech audio into text.

* llm_ttft (LLM time-to-first-token) :-
 - How long it took the GPT model to produce the first word after receiving your text.
    → Lower is better, shows model responsiveness.

* llm_total :-
 - Total time until GPT finished generating the entire reply.
    → This depends on how long the response is.

* tts_ttft_play (TTS time-to-first-token play) :-
 - How long from GPT finishing text to the first synthesized audio actually being played.
    → This measures Kokoro TTS synthesis + playback scheduling.

* e2e_ttft (End-to-End time-to-first-token) :-
 - Time from you finishing speaking → first AI audio played.
    → This is the real “latency” you experience in a conversation.

* user_text_len :-
 - Number of characters in your transcribed input.
    → Longer utterances don’t affect latency much since Whisper works on audio chunks.

* assistant_len :-
 - Number of characters in GPT’s reply.
    → Longer replies = bigger llm_total but do not change llm_ttft.
'''