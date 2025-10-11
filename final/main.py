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
from config_loader import SpeechToSpeechConfig

# ====================== LOAD CONFIG ======================
cfg = SpeechToSpeechConfig(os.path.join(os.path.dirname(__file__), "config.yaml"))

# ====================== FLASK / SOCKET.IO ======================
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# ====================== CLIENTS ======================
client_openai = OpenAI(api_key=cfg.openai_api_key)
client_elevenlabs = ElevenLabs(api_key=cfg.elevenlabs_api_key)

# Warm up ElevenLabs to reduce first-call latency
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
        "You are a helpful assistant. Keep your responses concise (1–2 sentences) "
        "unless more detail is requested."
    )
}]

rms_values = []
last_user_text = ""
LLM_LOCK = threading.Lock()

STOP_SIGNAL = threading.Event()
TTS_PLAYING = False
RESPONSE_GEN_ID = 0
TTS_PLAY_RMS = 0.0
BACKGROUND_RMS = 0.02
TTS_LAST_START_TIME = 0.0

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


vad_state = {"noise_floor": 0.02}  # initial baseline

def is_speech_adaptive(audio_chunk, sensitivity=3.0):
    """
    Cross-platform adaptive VAD:
    - tracks smoothed noise floor
    - updates dynamically based on recent mic levels
    - works across OS/mic differences
    """
    rms = float(np.sqrt(np.mean(audio_chunk**2) + 1e-10))

    # update noise floor when signal is quiet
    if rms < vad_state["noise_floor"] * (sensitivity * 0.8):
        vad_state["noise_floor"] = 0.97 * vad_state["noise_floor"] + 0.03 * rms

    threshold = vad_state["noise_floor"] * sensitivity
    is_voice = rms > threshold

    # optional debug:
    # print(f"RMS={rms:.5f}, floor={vad_state['noise_floor']:.5f}, thr={threshold:.5f}, speech={is_voice}")
    return is_voice


def linear_resample_int16(x_int16, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x_int16
    x = x_int16.astype(np.float32)
    N = x.shape[0]
    t_src = np.linspace(0, 1, N, endpoint=False)
    N_dst = int(np.round(N * dst_sr / src_sr))
    t_dst = np.linspace(0, 1, N_dst, endpoint=False)
    y = np.interp(t_dst, t_src, x)
    return np.clip(y, -32768, 32767).astype(np.int16)


def looks_useful_text(t: str) -> bool:
    t = t.strip()
    if len(t) < 3:
        return False
    words = [w for w in t.split() if w.isalpha()]
    return not (len(words) <= 1 and len(t) <= 4)

# ====================== TTS WORKERS ======================
def tts_synth_worker():
    """Convert text into PCM audio via ElevenLabs."""
    current_gen = None
    started = False
    t0_pipeline = None
    t0_tts = None

    while True:
        try:
            item = tts_synth_q.get(timeout=0.1)
        except queue.Empty:
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
                play_q.put({
                    "type": "play_start",
                    "t0_tts": t0_tts,
                    "t0_pipeline": t0_pipeline,
                    "gen_id": current_gen
                })
                started = True

            try:
                byte_iter = client_elevenlabs.text_to_speech.convert(
                    voice_id=cfg.elevenlabs_voice_id,
                    model_id=cfg.elevenlabs_model,
                    text=item["text"],
                    output_format=f"pcm_{cfg.output_sample_rate}"
                )
                buf = bytearray()
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
            play_q.put({"type": "play_end", "t0_tts": t0_tts, "t0_pipeline": t0_pipeline})
            current_gen = None
            started = False
            tts_synth_q.task_done()
            continue

        tts_synth_q.task_done()


def tts_play_worker():
    global TTS_PLAYING, TTS_LAST_START_TIME, TTS_PLAY_RMS
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
            stream = sd.OutputStream(samplerate=device_sr, channels=1, dtype="int16", blocksize=1024)
            stream.start()

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
            TTS_LAST_START_TIME = time.time()
            play_q.task_done()
            continue

        if isinstance(item, dict) and item.get("type") == "play_end":
           now = time.time()
           if t0_tts is not None:
               latency_stats["tts_total_ms"] = int((now - t0_tts) * 1000)
           if t0_pipeline is not None:
               latency_stats["pipeline_total_ms"] = int((now - t0_pipeline) * 1000)

            # --- NEW: print latency summary ---
           print(
               f"[PERCEPTUAL LATENCY] "
               f"STT={latency_stats['stt_latency_ms']}ms | "
               f"LLM_first={latency_stats['llm_first_token_ms']}ms | "
               f"TTS_first={latency_stats['tts_first_audio_ms']}ms | "
               f"Pipeline_first={latency_stats['pipeline_first_audio_ms']}ms"
            )
           socketio.start_background_task(socketio.emit, 'latency_update', latency_stats)
           TTS_PLAYING = False
           play_q.task_done()
           continue
        x = item
        if cfg.volume != 1.0:
            x = (x.astype(np.float32) * cfg.volume).clip(-32768, 32767).astype(np.int16)
        if device_sr != cfg.output_sample_rate:
            x = linear_resample_int16(x, cfg.output_sample_rate, device_sr)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if not first_audio_emitted and t0_tts is not None:
            now = time.time()
            latency_stats["tts_first_audio_ms"] = int((now - t0_tts) * 1000)
            if t0_pipeline is not None:
                latency_stats["pipeline_first_audio_ms"] = int((now - t0_pipeline) * 1000)
            first_audio_emitted = True

        CHUNK = 512
        i = 0
        while i < len(x):
            if STOP_SIGNAL.is_set():
                TTS_PLAYING = False
                break
            j = min(i + CHUNK, len(x))
            playback_rms = float(np.sqrt(np.mean(x[i:j].astype(np.float32)**2) + 1e-10))
            TTS_PLAY_RMS = 0.8 * TTS_PLAY_RMS + 0.2 * playback_rms
            stream.write(x[i:j, :])
            i = j
        play_q.task_done()

# ====================== INTERRUPT ======================
def interrupt_response():
    global TTS_PLAYING, RESPONSE_GEN_ID
    STOP_SIGNAL.set()

    for q in (tts_synth_q, play_q):
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break

    TTS_PLAYING = False
    RESPONSE_GEN_ID += 1
    socketio.start_background_task(socketio.emit, 'assistant_stream', {'token': "Oh, sorry — you go ahead."})
    socketio.start_background_task(socketio.emit, 'status', {'message': "Response interrupted. Listening..."})
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
            print(f"[ASSISTANT] {full_response}\n{'-' * 30}")
            rms_values.clear()
        except Exception as e:
            print(f"[LLM error] {e}")
            rms_values.clear()

# ====================== CORE LOOP ======================
def live_transcribe_loop():
    global last_user_text, RESPONSE_GEN_ID, BACKGROUND_RMS
    socketio.emit('status', {'message': "Loading Whisper model..."})
    try:
        model = whisper.load_model(cfg.whisper_model)
    except Exception as e:
        socketio.emit('error', {'message': f'Failed to load Whisper model: {e}'})
        return

    import json
    import os

    PROFILE_PATH = "audio_profile.json"

    print("Calibrating background noise (3s)...")
    calibration_samples = []
    with sd.InputStream(
      samplerate=cfg.sample_rate,
      channels=1,
      blocksize=cfg.block_size,
      callback=audio_callback
    ):
      time.sleep(3.0)

    while not audio_q.empty():
      calibration_samples.append(audio_q.get())

    if calibration_samples:
       all_audio = np.concatenate(calibration_samples)
       # Median RMS resists spikes
       BACKGROUND_RMS = float(np.median(np.abs(all_audio)))
       BACKGROUND_RMS = max(BACKGROUND_RMS, 1e-4)  # prevent zero

       # Adaptive thresholds
       cfg.mic_rms_baseline = max(0.03, BACKGROUND_RMS * 2.5)
       cfg.bargein_multiplier = 4.0 if BACKGROUND_RMS < 0.02 else 5.0

       # Save calibration profile
       with open(PROFILE_PATH, "w") as f:
         json.dump({
            "baseline": cfg.mic_rms_baseline,
            "multiplier": cfg.bargein_multiplier,
            "background_rms": BACKGROUND_RMS
         }, f, indent=2)

       print(f"[Calibration] Adaptive baseline={cfg.mic_rms_baseline:.4f}, "
          f"multiplier={cfg.bargein_multiplier:.1f}, "
          f"background_rms={BACKGROUND_RMS:.4f}")
    else:
    # Load previous calibration if available
      if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            prof = json.load(f)
        cfg.mic_rms_baseline = prof.get("baseline", 0.03)
        cfg.bargein_multiplier = prof.get("multiplier", 4.0)
        BACKGROUND_RMS = prof.get("background_rms", 0.02)
        print(f"[Calibration] Loaded profile: "
              f"baseline={cfg.mic_rms_baseline:.4f}, "
              f"multiplier={cfg.bargein_multiplier:.1f}, "
              f"background_rms={BACKGROUND_RMS:.4f}")
      else:
        BACKGROUND_RMS = 0.02
        cfg.mic_rms_baseline = 0.03
        cfg.bargein_multiplier = 4.0
        print("[Calibration] Using default thresholds.")

    socketio.emit('status', {'message': "Listening..."})
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        with sd.InputStream(samplerate=cfg.sample_rate, channels=1, blocksize=cfg.block_size, callback=audio_callback):
            while True:
                socketio.sleep(0.01)
                try:
                    audio = audio_q.get_nowait()
                    if not is_speech_adaptive(audio, sensitivity=cfg.vad_sensitivity) and not TTS_PLAYING:
                        mic_rms = float(np.sqrt(np.mean(audio**2) + 1e-10))
                        BACKGROUND_RMS = 0.95 * BACKGROUND_RMS + 0.05 * mic_rms

                except queue.Empty:
                    continue

                # === Barge-in detection ===
                if TTS_PLAYING:
                    time_since_tts = time.time() - TTS_LAST_START_TIME
                    if time_since_tts < cfg.bargein_ignore_sec:
                        continue
                    mic_rms = float(np.sqrt(np.mean(audio**2) + 1e-10))
                    rms_thr = max(cfg.mic_rms_baseline, BACKGROUND_RMS * cfg.bargein_multiplier)
                    if mic_rms > rms_thr:
                        print(f"[BARGE-IN] Detected (mic_rms={mic_rms:.4f}, thr={rms_thr:.4f}) → interrupt")
                        interrupt_response()
                        continue
                    continue

                buffer = np.concatenate((buffer, audio))
                if is_speech_adaptive(audio, sensitivity=cfg.vad_sensitivity):
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
                            socketio.emit('transcript_update', {'text': text})
                            if not LLM_LOCK.locked():
                                RESPONSE_GEN_ID += 1
                                threading.Thread(target=gpt_stream_and_queue_tts, args=(text, pipeline_t0), daemon=True).start()
                            else:
                                socketio.emit('status', {'message': "LLM busy..."})
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
        threading.Thread(target=tts_synth_worker, daemon=True).start()
        threading.Thread(target=tts_play_worker, daemon=True).start()
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
    socketio.run(app, host=cfg.host, port=cfg.port, debug=cfg.debug)