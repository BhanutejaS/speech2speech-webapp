import os
import queue
import time
import difflib
import numpy as np
import sounddevice as sd
import whisper
from dotenv import load_dotenv
from openai import OpenAI
import threading
import random
import re
from collections import deque

# ============================== CONFIG ===============================
load_dotenv()

# Flask & Socket.IO setup
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# ElevenLabs SDK
try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    print("Warning: ElevenLabs library not found. TTS will fail unless installed.")
    ElevenLabs = None

# API Keys / Clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel fallback

client_openai = OpenAI(api_key=OPENAI_API_KEY)
if ELEVENLABS_API_KEY and ElevenLabs:
    client_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)
else:
    client_elevenlabs = None
    print("Warning: ElevenLabs client not initialized. Check API key and library.")

# Mic / STT
SAMPLE_RATE = 16000
BLOCK_SIZE = 512

# VAD / Segmentation (tuned for tiny.en)
RMS_HISTORY = 30
SILENCE_MULTIPLIER = 2.0
MIN_CHUNK_SEC = 1.0
PAUSE_TO_CUT_SEC = 1.2

# Whisper STT (tiny.en for speed)
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
MAX_HISTORY = 6  # cap rolling context (besides the system prompt)

# System prompt for concise responses
SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep your responses concise and brief. "
    "Aim for 1-2 sentences maximum unless more detail is specifically requested."
)

# ElevenLabs TTS
EL_SR = 24000  # ElevenLabs PCM stream rate

# Gap Fillers (Used only for initial delay now)
GAP_FILLERS = ["um...", "uh...", "hmm..."]
MIN_GAP_DELAY_SEC = 0.3  # Minimum pause before a filler (if selected)
MAX_GAP_DELAY_SEC = 0.8  # Maximum pause before a filler (if selected)
GAP_FILLER_PROBABILITY = 1  # 35% chance to insert one at the start of response

# Playback
VOLUME = 0.9  # 0..1, applied in int16 domain

# ============================== STATE ===============================
audio_q = queue.Queue()       # mic frames (float32 @16k)
tts_synth_q = queue.Queue()   # items to synthesize: {type: start/text/end, ...}
play_q = queue.Queue()        # items to play: np.int16 or markers

conversation_context = [{"role": "system", "content": SYSTEM_PROMPT}]
rms_values = []
last_user_text = ""
LLM_LOCK = threading.Lock()

STOP_SIGNAL = threading.Event()
TTS_PLAYING = False
RESPONSE_GEN_ID = 0

# master pause + shutdown controls
MIC_PAUSED = threading.Event()     # when set => we ignore mic/STT
SHUTDOWN_EVENT = threading.Event()  # when set => break loops and stop app

# Cancellation watermark — any packets with gen_id <= this are dropped
CANCEL_UP_TO_GEN = -1

# Hard-cut event to reset audio output stream on STOP
CUT_AUDIO_EVENT = threading.Event()

# Latency metrics (updated by various threads)
latency_stats = {
    "stt_latency_ms": None,
    "llm_first_token_ms": None,
    "llm_total_ms": None,
    "tts_first_audio_ms": None,
    "tts_total_ms": None,
    "pipeline_first_audio_ms": None,
    "pipeline_total_ms": None,
}

# ==== Echo suppression + STOP gating ====
RECENT_ASSISTANT_SENTENCES = deque(maxlen=20)  # rolling buffer of last spoken assistant sentences
LAST_TTS_END_TIME = 0.0
IGNORE_AFTER_TTS_SEC = 0.3  # snappier

MIC_MUTE_UNTIL = 0.0        # hard gate after STOP is pressed
MUTE_AFTER_STOP_SEC = 0.4   # snappier

# Micro-hold before starting a clause to give STOP a chance to land
PRE_TTS_HOLD_MS = 120  # 80–150ms is typical


def sentence_split(text: str):
    parts = re.split(r"(?<=[\.\!\?\:;])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


def record_assistant_sentences(text: str):
    for s in sentence_split(text):
        if 3 <= len(s) <= 240:
            RECENT_ASSISTANT_SENTENCES.append(s)


def is_probable_echo(transcribed: str) -> bool:
    """Compare user STT transcript with recent assistant sentences to suppress echoes."""
    t = transcribed.strip().lower()
    if not t:
        return False
    for s in RECENT_ASSISTANT_SENTENCES:
        ss = s.lower()
        if t in ss or ss in t:
            return True
    for s in RECENT_ASSISTANT_SENTENCES:
        ratio = difflib.SequenceMatcher(None, t, s.lower()).ratio()
        if ratio >= 0.82:
            return True
    return False


def _drain_queue(q: queue.Queue):
    """Drop all pending items from a queue."""
    try:
        while True:
            q.get_nowait()
            q.task_done()
    except queue.Empty:
        pass


# ============================== HELPERS ===============================
def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
    """Called from a separate thread for each audio chunk."""
    if status:
        print(f"Audio stream status: {status}")
    audio_q.put(indata[:, 0].copy())


def is_speech(audio_chunk: np.ndarray) -> bool:
    """Simple RMS-based Voice Activity Detection (VAD)."""
    rms = float(np.sqrt(np.mean(audio_chunk ** 2) + 1e-10))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold


def linear_resample_int16(x_int16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x_int16
    mono = x_int16.ndim == 1
    x = x_int16.astype(np.float32)
    n = x.shape[0]
    t_src = np.linspace(0.0, 1.0, n, endpoint=False)
    n_dst = int(np.round(n * dst_sr / src_sr))
    t_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    if mono:
        y = np.interp(t_dst, t_src, x)
    else:
        y_l = np.interp(t_dst, t_src, x[:, 0])
        y_r = np.interp(t_dst, t_src, x[:, 1])
        y = np.stack([y_l, y_r], axis=1)
    return np.clip(y, -32768, 32767).astype(np.int16)


def looks_useful_text(t: str) -> bool:
    t = t.strip()
    if len(t) < 3:
        return False
    words = [w for w in t.split() if any(c.isalpha() for c in w)]
    if len(words) <= 1 and (len(t) <= 4):
        return False
    return True


def get_random_gap_filler() -> str:
    """Returns a random gap filler or an empty string based on probability."""
    if random.random() < GAP_FILLER_PROBABILITY:
        return random.choice(GAP_FILLERS)
    return ""


# ============================== TTS: SYNTH ===============================
def el_synthesize_to_play_queue(text: str, gen_id: int):
    """
    Use ElevenLabs SDK to convert text -> PCM 24k int16 and push to play_q in small blocks.
    Stops immediately if STOP is set or this generation is canceled.
    """
    if client_elevenlabs is None:
        print("[ElevenLabs] TTS client not initialized. Skipping audio synthesis.")
        return

    try:
        byte_iter = client_elevenlabs.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ElevenLabs API Error] {e}")
        return

    buf = bytearray()
    chunk_frames = 1024  # smaller chunks for faster mid-sentence cuts
    for b in byte_iter:
        if STOP_SIGNAL.is_set() or (gen_id <= CANCEL_UP_TO_GEN):
            break
        if not b:
            continue
        buf.extend(b)
        usable = len(buf) // 2 * 2
        if usable == 0:
            continue
        pcm = buf[:usable]
        del buf[:usable]
        x = np.frombuffer(pcm, dtype="<i2")  # int16 mono @24k
        i = 0
        while i < x.size:
            if STOP_SIGNAL.is_set() or (gen_id <= CANCEL_UP_TO_GEN):
                i = x.size
                break
            j = min(i + chunk_frames, x.size)
            play_q.put(x[i:j])
            i = j


# ============================== WORKERS ===============================
def tts_synth_worker():
    """
    Consumes tts_synth_q items fed by the LLM thread.
    Produces to play_q: audio frames and markers.

    NOTE: The previous inter-clause gap filler logic has been removed here.
    The filler is now ONLY handled in gpt_stream_and_queue_tts for the initial delay.
    """
    global CANCEL_UP_TO_GEN
    current_gen = None
    started = False
    t0_pipeline = None
    t0_tts = None
    is_first_clause = True

    while True:
        # Graceful shutdown
        if SHUTDOWN_EVENT.is_set():
            break

        try:
            item = tts_synth_q.get(timeout=0.1)
        except queue.Empty:
            if STOP_SIGNAL.is_set():
                current_gen = None
                started = False
                is_first_clause = True
            continue

        if item is None:
            tts_synth_q.task_done()
            break

        if STOP_SIGNAL.is_set():
            current_gen = None
            started = False
            is_first_clause = True
            tts_synth_q.task_done()
            continue

        itype = item.get("type")
        gen_id = item.get("gen_id")

        if gen_id is not None and gen_id <= CANCEL_UP_TO_GEN:
            tts_synth_q.task_done()
            continue

        if itype == "start":
            current_gen = gen_id
            t0_pipeline = item["pipeline_t0"]
            t0_tts = None
            started = False
            is_first_clause = True
            tts_synth_q.task_done()
            continue

        if itype == "text":
            if current_gen is None or gen_id != current_gen:
                tts_synth_q.task_done()
                continue

            if not started:
                # Initial hold before speaking
                hold_until = time.time() + (PRE_TTS_HOLD_MS / 1000.0)
                while time.time() < hold_until:
                    if STOP_SIGNAL.is_set() or (gen_id <= CANCEL_UP_TO_GEN) or SHUTDOWN_EVENT.is_set():
                        break
                    time.sleep(0.01)
                if STOP_SIGNAL.is_set() or (gen_id <= CANCEL_UP_TO_GEN) or SHUTDOWN_EVENT.is_set():
                    tts_synth_q.task_done()
                    continue
                t0_tts = time.time()
                play_q.put(
                    {
                        "type": "play_start",
                        "t0_tts": t0_tts,
                        "t0_pipeline": t0_pipeline,
                        "gen_id": current_gen,
                    }
                )
                started = True

            if STOP_SIGNAL.is_set() or (gen_id <= CANCEL_UP_TO_GEN) or SHUTDOWN_EVENT.is_set():
                tts_synth_q.task_done()
                continue

            # Synthesize the text (can be a filler or main response)
            text_to_say = item["text"]
            el_synthesize_to_play_queue(text_to_say, current_gen)

            # Only used to control inter-clause fillers (now removed)
            is_first_clause = False

            tts_synth_q.task_done()
            continue

        if itype == "end":
            if current_gen is None or gen_id != current_gen:
                tts_synth_q.task_done()
                continue
            play_q.put(
                {
                    "type": "play_end",
                    "t0_tts": t0_tts,
                    "t0_pipeline": t0_pipeline,
                    "gen_id": current_gen,
                }
            )
            current_gen = None
            started = False
            t0_pipeline = None
            t0_tts = None
            is_first_clause = True
            tts_synth_q.task_done()
            continue

        tts_synth_q.task_done()


def tts_play_worker():
    """
    Persistent OutputStream writer with robust device handling.
    Emits latency metrics and sets a cooldown to reduce mic echo.
    Drops packets for canceled generations.
    Hard-cuts device buffer on STOP.
    """
    global TTS_PLAYING, LAST_TTS_END_TIME, CANCEL_UP_TO_GEN

    stream = None
    device_sr = None
    output_device = None
    first_audio_emitted = False
    t0_tts = None
    t0_pipeline = None

    def pick_output_device_and_sr():
        try:
            devs = sd.query_devices()
        except Exception:
            return None, 48000

        try:
            default_idx = sd.default.device[1]
            if default_idx is not None and default_idx >= 0:
                info = sd.query_devices(default_idx, "output")
                sr = int(info.get("default_samplerate", 48000) or 48000)
                if info["max_output_channels"] >= 1:
                    return default_idx, sr
        except Exception:
            pass

        for i, d in enumerate(devs):
            if d.get("max_output_channels", 0) >= 1:
                sr = int(d.get("default_samplerate", 48000) or 48000)
                return i, sr

        return None, 48000

    def open_stream():
        nonlocal stream, device_sr, output_device
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass
        stream = None

        output_device, device_sr = pick_output_device_and_sr()
        try:
            stream = sd.OutputStream(
                samplerate=device_sr,
                device=output_device,
                channels=1,
                dtype="int16",
                blocksize=1024,
            )
            stream.start()
        except Exception as e:
            try:
                stream = sd.OutputStream(
                    samplerate=44100,
                    device=None,
                    channels=1,
                    dtype="int16",
                    blocksize=1024,
                )
                stream.start()
                device_sr = 44100
            except Exception as e2:
                raise RuntimeError(f"Failed to open audio output: {e} / {e2}")

    def ensure_stream():
        if stream is None:
            open_stream()

    def hard_reset_stream():
        nonlocal stream
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass
        stream = None
        open_stream()

    try:
        ensure_stream()
        while True:
            if SHUTDOWN_EVENT.is_set():
                break

            if CUT_AUDIO_EVENT.is_set():
                hard_reset_stream()
                CUT_AUDIO_EVENT.clear()

            item = play_q.get()
            if item is None:
                play_q.task_done()
                break

            if isinstance(item, dict):
                g = item.get("gen_id")
                if g is not None and g <= CANCEL_UP_TO_GEN:
                    play_q.task_done()
                    continue

            if isinstance(item, dict) and item.get("type") == "play_start":
                t0_tts = item["t0_tts"]
                t0_pipeline = item["t0_pipeline"]
                first_audio_emitted = False
                TTS_PLAYING = True
                play_q.task_done()
                continue

            if isinstance(item, dict) and item.get("type") == "play_end":
                now = time.time()
                if t0_tts is not None:
                    latency_stats["tts_total_ms"] = int((now - t0_tts) * 1000)
                if t0_pipeline is not None:
                    latency_stats["pipeline_total_ms"] = int((now - t0_pipeline) * 1000)

                socketio.start_background_task(socketio.emit, "latency_update", latency_stats)
                TTS_PLAYING = False

                LAST_TTS_END_TIME = time.time()
                socketio.start_background_task(
                    socketio.emit, "status", {"message": "Waiting (Cool-down)..."}
                )
                time.sleep(0.3)
                socketio.start_background_task(socketio.emit, "status", {"message": "Listening..."})

                t0_tts = None
                t0_pipeline = None
                first_audio_emitted = False
                play_q.task_done()
                continue

            ensure_stream()
            x = item
            if not isinstance(x, np.ndarray) or x.size == 0:
                play_q.task_done()
                continue

            if VOLUME != 1.0:
                x = (x.astype(np.float32) * float(VOLUME)).clip(-32768, 32767).astype(np.int16)
            if device_sr != EL_SR:
                x = linear_resample_int16(x, EL_SR, device_sr)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            if not first_audio_emitted and t0_tts is not None:
                now = time.time()
                latency_stats["tts_first_audio_ms"] = int((now - t0_tts) * 1000)
                if t0_pipeline is not None:
                    latency_stats["pipeline_first_audio_ms"] = int((now - t0_pipeline) * 1000)
                first_audio_emitted = True

            i = 0
            chunk = 1024
            while i < len(x):
                if SHUTDOWN_EVENT.is_set():
                    break

                if CUT_AUDIO_EVENT.is_set():
                    CUT_AUDIO_EVENT.clear()
                    hard_reset_stream()
                    break

                if STOP_SIGNAL.is_set():
                    TTS_PLAYING = False
                    while not play_q.empty():
                        try:
                            drop = play_q.get_nowait()
                            play_q.task_done()
                        except queue.Empty:
                            break
                    hard_reset_stream()
                    break

                j = min(i + chunk, len(x))
                try:
                    stream.write(x[i:j, :])
                except Exception:
                    open_stream()
                    stream.write(x[i:j, :])
                i = j

            play_q.task_done()
    except Exception as e:  # noqa: BLE001
        print(f"[TTS play worker error] {e}")
        socketio.start_background_task(socketio.emit, "error", {"message": f"TTS Playback Error: {e}"})
    finally:
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass


# ============================== INTERRUPT / PAUSE / SHUTDOWN ===============================
def interrupt_response():
    """Immediately stops LLM streaming and clears all outgoing queues."""
    global TTS_PLAYING, RESPONSE_GEN_ID, MIC_MUTE_UNTIL, LAST_TTS_END_TIME, CANCEL_UP_TO_GEN, rms_values

    STOP_SIGNAL.set()
    CUT_AUDIO_EVENT.set()  # force playback stream reset

    RESPONSE_GEN_ID += 1
    CANCEL_UP_TO_GEN = RESPONSE_GEN_ID  # watermark for workers

    _drain_queue(tts_synth_q)
    _drain_queue(play_q)
    _drain_queue(audio_q)

    TTS_PLAYING = False
    MIC_MUTE_UNTIL = time.time() + MUTE_AFTER_STOP_SEC
    LAST_TTS_END_TIME = MIC_MUTE_UNTIL
    rms_values.clear()

    time.sleep(0.1)

    socketio.start_background_task(
        socketio.emit, "status", {"message": "Response interrupted. Listening..."}
    )
    print("[INTERRUPT] Response stopped by user.")


def pause_listening():
    """Pause mic + STT until resumed."""
    MIC_PAUSED.set()
    _drain_queue(audio_q)
    socketio.start_background_task(socketio.emit, "status", {"message": "Listening paused."})
    print("[MIC] Listening paused.")


def resume_listening():
    """Resume mic + STT."""
    MIC_PAUSED.clear()
    socketio.start_background_task(socketio.emit, "status", {"message": "Listening..."})
    print("[MIC] Listening resumed.")


def shutdown_app():
    """Gracefully shut down the entire application."""
    print("[SHUTDOWN] Requested by client. Shutting down...")
    SHUTDOWN_EVENT.set()

    # Stop any current response and playback
    try:
        interrupt_response()
    except Exception:
        pass

    # Ask workers to exit
    try:
        tts_synth_q.put(None)
        play_q.put(None)
    except Exception:
        pass

    # Notify clients, then stop server
    socketio.emit("app_shutdown", {"message": "Bye… See you 👋"})
    time.sleep(0.4)
    # Aggressive exit for clean Flask/SocketIO shutdown
    os._exit(0)


# Start workers upon server start
threading.Thread(target=tts_synth_worker, daemon=True).start()
threading.Thread(target=tts_play_worker, daemon=True).start()

# ============================== LLM ===============================
# ============================== LLM (DEFINITIVE FIX) ===============================
def gpt_stream_and_queue_tts(segment_text: str, pipeline_t0: float):
    """
    Single LLM pass per user segment.
    INSERTS GAP FILLER BEFORE LLM STREAMING STARTS to mask initial latency.
    """
    global conversation_context, RESPONSE_GEN_ID, rms_values, CANCEL_UP_TO_GEN

    with LLM_LOCK:
        if SHUTDOWN_EVENT.is_set():
            return

        my_gen = RESPONSE_GEN_ID

        conversation_context.append({"role": "user", "content": segment_text})
        if len(conversation_context) > (2 * MAX_HISTORY + 1):
            conversation_context[:] = [conversation_context[0]] + conversation_context[-(2 * MAX_HISTORY):]

        print(f"\n\n[USER] {segment_text}")

        full_response = ""
        aborted = False
        sent_start_marker = False
        
        # --- DEFINITIVE GAP FILLER LOGIC ---
        filler = get_random_gap_filler()
        if filler and client_elevenlabs is not None:
            # 1. Send 'start' marker first
            tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
            sent_start_marker = True
            
            # 2. Insert the filler text BEFORE starting LLM stream processing
            print(f"[TTS-SYNTH] Inserted GUARANTEED INITIAL filler: {filler}")
            tts_synth_q.put({"type": "text", "text": filler, "gen_id": my_gen})
            record_assistant_sentences(filler) 
            
        # --- END DEFINITIVE GAP FILLER LOGIC ---

        try:
            resp = client_openai.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True,
            )

            t_llm_start = time.time()
            first_token_time = None
            clause_buf = ""

            for chunk in resp:
                if SHUTDOWN_EVENT.is_set() or STOP_SIGNAL.is_set() or my_gen <= CANCEL_UP_TO_GEN:
                    aborted = True
                    break

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    token = delta.content

                    if SHUTDOWN_EVENT.is_set() or STOP_SIGNAL.is_set() or my_gen <= CANCEL_UP_TO_GEN:
                        aborted = True
                        break

                    full_response += token
                    clause_buf += token

                    socketio.emit("assistant_stream", {"token": token})

                    if first_token_time is None:
                        first_token_time = time.time()
                        latency_stats["llm_first_token_ms"] = int((first_token_time - t_llm_start) * 1000)

                    end_of_clause = any(p in token for p in ".?!;:")
                    if end_of_clause and len(clause_buf.strip()) > 2:
                        
                        # Use the start marker if it wasn't sent for the initial filler
                        if not sent_start_marker:
                            tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
                            sent_start_marker = True

                        if SHUTDOWN_EVENT.is_set() or STOP_SIGNAL.is_set() or my_gen <= CANCEL_UP_TO_GEN:
                            aborted = True
                            break

                        text_to_say = clause_buf.strip()
                        tts_synth_q.put({"type": "text", "text": text_to_say, "gen_id": my_gen})
                        record_assistant_sentences(text_to_say)
                        clause_buf = ""

            if not aborted and clause_buf.strip():
                if not sent_start_marker:
                    tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
                    sent_start_marker = True
                text_to_say = clause_buf.strip()
                tts_synth_q.put({"type": "text", "text": text_to_say, "gen_id": my_gen})
                record_assistant_sentences(text_to_say)

            if sent_start_marker and not aborted:
                tts_synth_q.put({"type": "end", "gen_id": my_gen})

            latency_stats["llm_total_ms"] = int((time.time() - t_llm_start) * 1000)

            if not aborted:
                conversation_context.append({"role": "assistant", "content": full_response})
                print(f"[ASSISTANT] {full_response}")
                print("-" * 30)
            else:
                if conversation_context and conversation_context[-1]["role"] == "user":
                    conversation_context.pop()

            rms_values.clear()

        except Exception as e: # noqa: BLE001
            print(f"\n[LLM error] {e}")
            socketio.start_background_task(socketio.emit, "error", {"message": f"LLM Generation Error: {e}"})
            if conversation_context and conversation_context[-1]["role"] == "user":
                conversation_context.pop()
            rms_values.clear()
# ============================== CORE LOOP ===============================
def live_transcribe_loop():
    """
    Mic -> VAD -> STT (tiny.en) -> LLM -> TTS.
    Runs in a dedicated thread.
    """
    global last_user_text, RESPONSE_GEN_ID, MIC_MUTE_UNTIL

    socketio.emit("status", {"message": "Loading Whisper model..."})
    try:
        model = whisper.load_model(WHISPER_MODEL)
    except Exception as e:  # noqa: BLE001
        socketio.start_background_task(socketio.emit, "error", {"message": f"Failed to load Whisper model: {e}"})
        print(f"Failed to load Whisper model: {e}")
        return

    socketio.emit("status", {"message": "Listening..."})
    print("Listening... (Ctrl+C to exit)")

    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                if SHUTDOWN_EVENT.is_set():
                    break

                socketio.sleep(0.01)

                # If listening is paused, keep UI updated and ignore audio completely
                if MIC_PAUSED.is_set():
                    _drain_queue(audio_q)
                    status_now = "Listening paused."
                    socketio.emit("status", {"message": status_now})
                    continue

                # Auto-clear STOP once the mute window passes
                if STOP_SIGNAL.is_set() and time.time() >= MIC_MUTE_UNTIL:
                    STOP_SIGNAL.clear()
                    print("[MIC] Re-enabled after stop/mute window")

                if STOP_SIGNAL.is_set():
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                try:
                    audio = audio_q.get_nowait()
                except queue.Empty:
                    if time.time() < MIC_MUTE_UNTIL:
                        continue
                    continue

                # 1) Don't process while the assistant is talking
                if TTS_PLAYING:
                    continue

                # 2) Ignore mic for a window after playback OR stop
                if (time.time() - LAST_TTS_END_TIME) < IGNORE_AFTER_TTS_SEC or (time.time() < MIC_MUTE_UNTIL):
                    continue

                buffer = np.concatenate((buffer, audio))

                if is_speech(audio):
                    last_speech_time = time.time()

                enough_audio = buffer.size >= int(MIN_CHUNK_SEC * SAMPLE_RATE)
                paused_long = last_speech_time and (time.time() - last_speech_time > PAUSE_TO_CUT_SEC)

                if enough_audio and paused_long:
                    stt_t0 = time.time()
                    pipeline_t0 = stt_t0

                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                    socketio.emit("status", {"message": "Transcribing audio..."})

                    print(f"\n[STT] Transcribing {len(audio_chunk) / SAMPLE_RATE:.2f}s...")
                    result = model.transcribe(audio_chunk, **WHISPER_ARGS)
                    text = (result.get("text") or "").strip()
                    latency_stats["stt_latency_ms"] = int((time.time() - stt_t0) * 1000)

                    if text:
                        if is_probable_echo(text):
                            print(f"[STT] Suppressed echo of assistant: {text!r}")
                            socketio.emit("status", {"message": "Listening..."})
                            continue

                        if not looks_useful_text(text):
                            print(f"[STT] Ignored short/low-content: {text!r}")
                            socketio.emit("status", {"message": "Listening..."})
                            continue
                        sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                        if sim > 0.85 or text.lower() in last_user_text.lower():
                            print(f"[STT] Suppressed duplicate user text: {text!r}")
                            socketio.emit("status", {"message": "Listening..."})
                            continue
                        last_user_text = text

                        socketio.emit("transcript_update", {"text": text, "t_stt": latency_stats["stt_latency_ms"]})

                        if not LLM_LOCK.locked():
                            RESPONSE_GEN_ID += 1
                            threading.Thread(
                                target=gpt_stream_and_queue_tts,
                                args=(text, pipeline_t0),
                                daemon=True,
                            ).start()
                        else:
                            socketio.emit("status", {"message": "LLM busy, waiting for next turn..."})

    except Exception as e:  # noqa: BLE001
        print(f"[Core error] {e}")
        socketio.start_background_task(socketio.emit, "error", {"message": f"Core Loop Error: {e}"})
    finally:
        socketio.emit("status", {"message": "Assistant stopped."})


# ============================== FLASK ROUTES & SOCKET.IO EVENTS ===============================
@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("status", {"message": "Connected. Initializing audio stream..."})
    if not hasattr(socketio, "assistant_thread"):
        # Ensure 'templates' directory exists for Flask
        if not os.path.exists("templates"):
            os.makedirs("templates")

        # Start the core loop thread only once
        socketio.assistant_thread = socketio.start_background_task(live_transcribe_loop)


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("stop_response")
def handle_stop_response():
    threading.Thread(target=interrupt_response, daemon=True).start()


# pause / resume / shutdown events
@socketio.on("pause_listening")
def handle_pause_listening():
    pause_listening()


@socketio.on("resume_listening")
def handle_resume_listening():
    resume_listening()


@socketio.on("shutdown_app")
def handle_shutdown():
    shutdown_app()


# ============================== MAIN ===============================
if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")

    print("Starting AI Voice Assistant server...")
    # allow_unsafe_werkzeug=True is needed for some environments/Flask versions
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)
