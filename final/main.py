import os
import queue
import threading
import time
import difflib
import numpy as np
import sounddevice as sd
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from elevenlabs.client import ElevenLabs

# === Custom config loader ===
from config_loader import SpeechToSpeechConfig

# ====================== LOAD CONFIG ======================
cfg = SpeechToSpeechConfig("config.yaml")

# ====================== FLASK / SOCKET.IO ======================
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# ====================== CLIENTS ======================
client_openai = OpenAI(api_key=cfg.openai_api_key)
client_elevenlabs = ElevenLabs(api_key=cfg.elevenlabs_api_key)

# Warm up ElevenLabs
try:
    _ = client_elevenlabs.text_to_speech.convert(
        voice_id=cfg.elevenlabs_voice_id,
        text=".",
        model_id=cfg.elevenlabs_model,
        output_format=f"pcm_{cfg.output_sample_rate}"
    )
except Exception:
    pass

# ====================== STATE ======================
audio_q = queue.Queue()
tts_synth_q = queue.Queue()
play_q = queue.Queue()

conversation_context = [{
    "role": "system",
    "content": (
        "You are a helpful assistant. Keep responses concise (1–2 sentences)."
    )
}]

rms_values = []
last_user_text = ""
LLM_LOCK = threading.Lock()

STOP_SIGNAL = threading.Event()
TTS_PLAYING = False
RESPONSE_GEN_ID = 0
TTS_PLAY_RMS = 0.0

latency_stats = {
    "stt_latency_ms": None,
    "llm_first_token_ms": None,
    "llm_total_ms": None,
    "tts_first_audio_ms": None,
    "tts_total_ms": None,
    "pipeline_first_audio_ms": None,
    "pipeline_total_ms": None,
}

# ====================== HELPERS ======================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_q.put(indata[:, 0].copy())

def is_speech(audio_chunk):
    rms = float(np.sqrt(np.mean(audio_chunk ** 2) + 1e-10))
    rms_values.append(rms)
    if len(rms_values) > cfg.rms_history:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * cfg.silence_multiplier
    return rms > threshold

def linear_resample_int16(x_int16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x_int16
    x = x_int16.astype(np.float32)
    N = x.shape[0]
    t_src = np.linspace(0.0, 1.0, N, endpoint=False)
    N_dst = int(np.round(N * dst_sr / src_sr))
    t_dst = np.linspace(0.0, 1.0, N_dst, endpoint=False)
    y = np.interp(t_dst, t_src, x)
    return np.clip(y, -32768, 32767).astype(np.int16)

def looks_useful_text(t: str) -> bool:
    t = t.strip()
    if len(t) < 3:
        return False
    words = [w for w in t.split() if w.isalpha()]
    if len(words) <= 1 and len(t) <= 4:
        return False
    return True

# ====================== TTS WORKER ======================
def tts_synth_worker():
    """ElevenLabs REST API TTS (non-realtime)."""
    current_gen = None
    started = False
    t0_pipeline = None
    t0_tts = None

    while True:
        try:
            item = tts_synth_q.get(timeout=0.1)
        except queue.Empty:
            if STOP_SIGNAL.is_set():
                continue
            continue

        if item is None:
            tts_synth_q.task_done()
            break

        if STOP_SIGNAL.is_set():
            tts_synth_q.task_done()
            continue

        itype = item.get("type")
        gen_id = item.get("gen_id")

        if itype == "start":
            current_gen = gen_id
            t0_pipeline = item["pipeline_t0"]
            started = False
            tts_synth_q.task_done()
            continue

        if itype == "text":
            if current_gen is None or gen_id != current_gen:
                tts_synth_q.task_done()
                continue
            if not started:
                t0_tts = time.time()
                play_q.put({"type": "play_start", "t0_tts": t0_tts, "t0_pipeline": t0_pipeline, "gen_id": current_gen})
                started = True

            text = item["text"]
            try:
                byte_iter = client_elevenlabs.text_to_speech.convert(
                    voice_id=cfg.elevenlabs_voice_id,
                    model_id=cfg.elevenlabs_model,
                    text=text,
                    output_format=f"pcm_{cfg.output_sample_rate}"
                )

                buf = bytearray()
                CHUNK_FRAMES = 2048
                for b in byte_iter:
                    if STOP_SIGNAL.is_set():
                        break
                    if not b:
                        continue
                    buf.extend(b)
                    usable = len(buf) // 2 * 2
                    if usable == 0:
                        continue
                    pcm = buf[:usable]
                    del buf[:usable]
                    x = np.frombuffer(pcm, dtype="<i2")
                    play_q.put(x)
            except Exception as e:
                print(f"[ElevenLabs API error] {e}")

            tts_synth_q.task_done()
            continue

        if itype == "end":
            if current_gen is None or gen_id != current_gen:
                tts_synth_q.task_done()
                continue
            play_q.put({"type": "play_end", "t0_tts": t0_tts, "t0_pipeline": t0_pipeline, "gen_id": current_gen})
            current_gen = None
            started = False
            t0_pipeline = None
            t0_tts = None
            tts_synth_q.task_done()
            continue

        tts_synth_q.task_done()

# ====================== PLAYBACK WORKER ======================
def tts_play_worker():
    global TTS_PLAYING, TTS_PLAY_RMS
    stream = None
    device_sr = None
    first_audio_emitted = False
    t0_tts = None
    t0_pipeline = None

    def ensure_stream():
        nonlocal stream, device_sr
        if stream is None:
            try:
                dev = sd.query_devices(None, "output")
                device_sr = int(dev.get("default_samplerate", 48000))
            except Exception:
                device_sr = 48000
            stream = sd.OutputStream(
                samplerate=device_sr,
                channels=1,
                dtype="int16",
                blocksize=cfg.play_chunk,
            )
            stream.start()

    try:
        ensure_stream()
        while True:
            item = play_q.get()
            if item is None:
                play_q.task_done()
                break

            if isinstance(item, dict) and item.get("type") == "play_start":
                t0_tts = item["t0_tts"]
                t0_pipeline = item["t0_pipeline"]
                first_audio_emitted = False
                TTS_PLAYING = True
                play_q.task_done()
                continue

            if isinstance(item, dict) and item.get("type") == "play_end":
                now = time.time()
                if t0_tts:
                    latency_stats["tts_total_ms"] = int((now - t0_tts) * 1000)
                if t0_pipeline:
                    latency_stats["pipeline_total_ms"] = int((now - t0_pipeline) * 1000)
                print_latency()
                TTS_PLAYING = False
                t0_tts = None
                t0_pipeline = None
                first_audio_emitted = False
                play_q.task_done()
                continue

            ensure_stream()
            x = item
            if cfg.volume != 1.0:
                x = (x.astype(np.float32) * cfg.volume).clip(-32768, 32767).astype(np.int16)
            if device_sr != cfg.output_sample_rate:
                x = linear_resample_int16(x, cfg.output_sample_rate, device_sr)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            if not first_audio_emitted and t0_tts:
                now = time.time()
                latency_stats["tts_first_audio_ms"] = int((now - t0_tts) * 1000)
                latency_stats["pipeline_first_audio_ms"] = int((now - t0_pipeline) * 1000)
                first_audio_emitted = True

            i = 0
            while i < len(x):
                if STOP_SIGNAL.is_set():
                    TTS_PLAYING = False
                    break
                j = min(i + cfg.play_chunk, len(x))
                playback_rms = float(np.sqrt(np.mean(x[i:j].astype(np.float32) ** 2) + 1e-10))
                TTS_PLAY_RMS = 0.8 * TTS_PLAY_RMS + 0.2 * playback_rms
                stream.write(x[i:j, :])
                i = j
            play_q.task_done()
    except Exception as e:
        print(f"[TTS playback error] {e}")
    finally:
        try:
            if stream is not None:
                stream.stop(); stream.close()
        except Exception:
            pass

def print_latency():
    print(
        f"[LATENCY] STT={latency_stats['stt_latency_ms']}ms | "
        f"LLM_first={latency_stats['llm_first_token_ms']}ms | "
        f"LLM_total={latency_stats['llm_total_ms']}ms | "
        f"TTS_first={latency_stats['tts_first_audio_ms']}ms | "
        f"Pipeline_first={latency_stats['pipeline_first_audio_ms']}ms | "
        f"Pipeline_total={latency_stats['pipeline_total_ms']}ms"
    )

# ====================== INTERRUPT ======================
def interrupt_response():
    global TTS_PLAYING, RESPONSE_GEN_ID
    STOP_SIGNAL.set()
    while not tts_synth_q.empty():
        try:
            tts_synth_q.get_nowait(); tts_synth_q.task_done()
        except queue.Empty:
            break
    while not play_q.empty():
        try:
            play_q.get_nowait(); play_q.task_done()
        except queue.Empty:
            break
    TTS_PLAYING = False
    RESPONSE_GEN_ID += 1
    socketio.start_background_task(socketio.emit, 'assistant_stream', {'token': "Oh, sorry — you go ahead."})
    print("[INTERRUPT] Response stopped.")

# ====================== LLM STREAMING ======================
def gpt_stream_and_queue_tts(segment_text: str, pipeline_t0: float):
    global conversation_context, RESPONSE_GEN_ID, rms_values
    with LLM_LOCK:
        STOP_SIGNAL.clear()
        my_gen = RESPONSE_GEN_ID

        conversation_context.append({"role": "user", "content": segment_text})
        if len(conversation_context) > (2 * cfg.max_history + 1):
            conversation_context[:] = [conversation_context[0]] + conversation_context[-(2 * cfg.max_history):]

        print(f"\n[USER] {segment_text}")
        full_response = ""
        try:
            resp = client_openai.chat.completions.create(
                model=cfg.llm_model,
                messages=conversation_context,
                stream=True,
                max_completion_tokens=50
            )

            t_llm_start = time.time()
            first_token_time = None
            clause_buf = ""
            sent_start_marker = False

            for chunk in resp:
                if STOP_SIGNAL.is_set():
                    print("[LLM] Interrupted.")
                    if conversation_context[-1]["role"] == "user":
                        conversation_context.pop()
                    return

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    token = delta.content
                    full_response += token
                    clause_buf += token
                    socketio.emit('assistant_stream', {'token': token})

                    if first_token_time is None:
                        first_token_time = time.time()
                        latency_stats["llm_first_token_ms"] = int((first_token_time - t_llm_start) * 1000)

                    if len(clause_buf.split()) >= 3 or any(p in token for p in ".?!,;:"):
                        if not sent_start_marker:
                            tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
                            sent_start_marker = True
                        tts_synth_q.put({"type": "text", "text": clause_buf.strip(), "gen_id": my_gen})
                        clause_buf = ""

            if clause_buf.strip():
                if not sent_start_marker:
                    tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
                tts_synth_q.put({"type": "text", "text": clause_buf.strip(), "gen_id": my_gen})
            if sent_start_marker:
                tts_synth_q.put({"type": "end", "gen_id": my_gen})

            latency_stats["llm_total_ms"] = int((time.time() - t_llm_start) * 1000)
            conversation_context.append({"role": "assistant", "content": full_response})
            print(f"[ASSISTANT] {full_response}")
            print("-" * 30)
            rms_values.clear()
        except Exception as e:
            print(f"[LLM error] {e}")
            rms_values.clear()

# ====================== CORE LOOP ======================
def live_transcribe_loop():
    global last_user_text, RESPONSE_GEN_ID
    socketio.emit('status', {'message': "Loading Whisper model..."})
    try:
        model = whisper.load_model(cfg.whisper_model)
    except Exception as e:
        socketio.emit('error', {'message': f'Failed to load Whisper: {e}'})
        return

    socketio.emit('status', {'message': "Listening..."})
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        with sd.InputStream(samplerate=cfg.sample_rate, channels=1, blocksize=cfg.block_size, callback=audio_callback):
            while True:
                socketio.sleep(0.01)
                try:
                    audio = audio_q.get_nowait()
                except queue.Empty:
                    continue
                if TTS_PLAYING:
                    continue
                buffer = np.concatenate((buffer, audio))
                if is_speech(audio):
                    last_speech_time = time.time()
                enough_audio = buffer.size >= int(cfg.min_chunk_sec * cfg.sample_rate)
                paused_long = last_speech_time and (time.time() - last_speech_time > cfg.pause_to_cut_sec)
                if enough_audio and paused_long:
                    stt_t0 = time.time()
                    pipeline_t0 = stt_t0
                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None
                    socketio.emit('status', {'message': "Transcribing..."})
                    result = model.transcribe(audio_chunk, fp16=False, language="en", temperature=0.0)
                    text = (result.get("text") or "").strip()
                    latency_stats["stt_latency_ms"] = int((time.time() - stt_t0) * 1000)
                    if text and looks_useful_text(text):
                        sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                        if sim < 0.85:
                            last_user_text = text
                            socketio.emit('transcript_update', {'text': text, 't_stt': latency_stats["stt_latency_ms"]})
                            if not LLM_LOCK.locked():
                                RESPONSE_GEN_ID += 1
                                threading.Thread(target=gpt_stream_and_queue_tts, args=(text, pipeline_t0), daemon=True).start()
    except Exception as e:
        print(f"[Core error] {e}")

# ====================== FLASK ROUTES ======================
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print('Client connected')
    emit('status', {'message': 'Connected. Listening...'})
    if not hasattr(socketio, 'assistant_thread'):
        socketio.assistant_thread = socketio.start_background_task(live_transcribe_loop)

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('stop_response')
def on_stop_response():
    threading.Thread(target=interrupt_response, daemon=True).start()

# ====================== MAIN ======================
if __name__ == "__main__":
    print(f"Starting AI Voice Assistant on {cfg.host}:{cfg.port} ...")
    threading.Thread(target=tts_synth_worker, daemon=True).start()
    threading.Thread(target=tts_play_worker, daemon=True).start()
    socketio.run(app, host=cfg.host, port=cfg.port, debug=cfg.debug)
