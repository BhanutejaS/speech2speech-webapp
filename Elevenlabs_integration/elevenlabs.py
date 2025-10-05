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

# Flask and Socket.IO for the web interface
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, disconnect

# ElevenLabs SDK
from elevenlabs.client import ElevenLabs

# ============================== CONFIG ===============================
load_dotenv()

# Flask & Socket.IO setup
app = Flask(__name__)
# Suppress the default Flask logger messages if desired
# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)
socketio = SocketIO(app, async_mode='threading')

# API Keys / Clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel fallback

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Mic / STT
SAMPLE_RATE = 16000
BLOCK_SIZE = 512

# VAD / Segmentation (tuned for tiny.en)
RMS_HISTORY = 30
SILENCE_MULTIPLIER = 2.0     # slightly stricter than before
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

# ElevenLabs TTS
EL_SR = 24000  # ElevenLabs PCM stream rate

# Playback
VOLUME = 0.9  # 0..1, applied in int16 domain

# ============================== STATE ===============================
audio_q = queue.Queue()       # mic frames (float32 @16k)
tts_synth_q = queue.Queue() # items to synthesize: {type: start/text/end, ...}
play_q = queue.Queue()        # items to play: np.int16 or markers

conversation_context = [{"role": "system", "content": "You are a helpful assistant."}]
rms_values = []
last_user_text = ""
LLM_LOCK = threading.Lock()

STOP_SIGNAL = threading.Event()
TTS_PLAYING = False
RESPONSE_GEN_ID = 0

# Latency metrics
# Stored globally and updated by various threads
latency_stats = {
    "stt_latency_ms": None,
    "llm_first_token_ms": None,
    "llm_total_ms": None,
    "tts_first_audio_ms": None,
    "tts_total_ms": None,
    "pipeline_first_audio_ms": None,
    "pipeline_total_ms": None,
}

# ============================== HELPERS ===============================
def audio_callback(indata, frames, time_info, status):
    """Called from a separate thread for each audio chunk."""
    if status:
        print(f"Audio stream status: {status}")
    audio_q.put(indata[:, 0].copy())

def is_speech(audio_chunk):
    """Simple RMS-based Voice Activity Detection (VAD)."""
    rms = float(np.sqrt(np.mean(audio_chunk**2) + 1e-10))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values) if rms_values else 0.0
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

def linear_resample_int16(x_int16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x_int16
    mono = (x_int16.ndim == 1)
    x = x_int16.astype(np.float32)
    N = x.shape[0]
    t_src = np.linspace(0.0, 1.0, N, endpoint=False)
    N_dst = int(np.round(N * dst_sr / src_sr))
    t_dst = np.linspace(0.0, 1.0, N_dst, endpoint=False)
    if mono:
        y = np.interp(t_dst, t_src, x)
    else:
        yL = np.interp(t_dst, t_src, x[:, 0])
        yR = np.interp(t_dst, t_src, x[:, 1])
        y = np.stack([yL, yR], axis=1)
    return np.clip(y, -32768, 32767).astype(np.int16)

def looks_useful_text(t: str) -> bool:
    t = t.strip()
    if len(t) < 3:
        return False
    words = [w for w in t.split() if w.isalpha()]
    if len(words) <= 1 and (len(t) <= 4):
        return False
    return True

# ============================== TTS: SYNTH ===============================
def el_synthesize_to_play_queue(text: str):
    """
    Use ElevenLabs SDK to convert text -> PCM 24k int16 and push to play_q in small blocks.
    """
    try:
        byte_iter = client_elevenlabs.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        )
    except Exception as e:
        print(f"[ElevenLabs API Error] {e}")
        return

    buf = bytearray()
    CHUNK_FRAMES = 2048
    for b in byte_iter:
        if STOP_SIGNAL.is_set():
            return
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
            j = min(i + CHUNK_FRAMES, x.size)
            play_q.put(x[i:j])
            i = j

# ============================== WORKERS ===============================
def tts_synth_worker():
    """
    Consumes tts_synth_q items fed by the LLM thread.
    Produces to play_q: audio frames and markers.
    """
    current_gen = None
    started = False
    t0_pipeline = None
    t0_tts = None

    while True:
        try:
            # Use a timeout to allow the thread to check STOP_SIGNAL periodically
            item = tts_synth_q.get(timeout=0.1) 
        except queue.Empty:
            if STOP_SIGNAL.is_set():
                # If we're interrupted and the queue is empty, we just loop again
                continue
            continue
            
        # If interrupted, drop the item and loop again (interrupt_response clears the queue)
        if STOP_SIGNAL.is_set(): 
            tts_synth_q.task_done()
            continue

        if item is None:
            tts_synth_q.task_done()
            break

        itype = item.get("type")
        gen_id = item.get("gen_id")

        if itype == "start":
            current_gen = gen_id
            t0_pipeline = item["pipeline_t0"]
            t0_tts = None
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
            el_synthesize_to_play_queue(item["text"])
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

def tts_play_worker():
    """
    Persistent OutputStream writer. Calculates and emits latency metrics.
    """
    global TTS_PLAYING
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
                blocksize=1024,
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
                if t0_tts is not None:
                    latency_stats["tts_total_ms"] = int((now - t0_tts) * 1000)
                if t0_pipeline is not None:
                    latency_stats["pipeline_total_ms"] = int((now - t0_pipeline) * 1000)
                
                # EMIT FINAL LATENCY STATS
                # Use socketio.start_background_task for emission outside the main event loop
                socketio.start_background_task(socketio.emit, 'latency_update', latency_stats)
                
                TTS_PLAYING = False
                t0_tts = None
                t0_pipeline = None
                first_audio_emitted = False
                play_q.task_done()
                continue

            ensure_stream()
            x = item
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
            CHUNK = 2048
            while i < len(x):
                if STOP_SIGNAL.is_set():
                    TTS_PLAYING = False # Stop playback immediately if interrupted
                    # Clear out remaining audio in play_q for this generation
                    while not play_q.empty():
                        try:
                            item_to_drop = play_q.get_nowait()
                            if isinstance(item_to_drop, dict) and item_to_drop.get("type") == "play_end":
                                # Don't drop the 'play_end' marker, let it run next time
                                play_q.put(item_to_drop)
                                break
                            play_q.task_done()
                        except queue.Empty:
                            break
                    break # Exit the while i < len(x) loop
                j = min(i + CHUNK, len(x))
                stream.write(x[i:j, :])
                i = j
            play_q.task_done()
    except Exception as e:
        print(f"[TTS play worker error] {e}")
        socketio.start_background_task(socketio.emit, 'error', {'message': f'TTS Playback Error: {e}'})
    finally:
        try:
            if stream is not None:
                stream.stop(); stream.close()
        except Exception:
            pass

# ============================== INTERRUPT ===============================
def interrupt_response():
    """Immediately stops LLM streaming and clears all outgoing queues."""
    global TTS_PLAYING, RESPONSE_GEN_ID
    
    # 1. Set the global stop signal
    STOP_SIGNAL.set()
    
    # 2. Clear all outgoing queues to immediately stop processing new items
    while not tts_synth_q.empty():
        try:
            tts_synth_q.get_nowait()
            tts_synth_q.task_done()
        except queue.Empty:
            break
    
    while not play_q.empty():
        try:
            play_q.get_nowait()
            play_q.task_done()
        except queue.Empty:
            break

    # 3. Reset state variables
    TTS_PLAYING = False
    RESPONSE_GEN_ID += 1 # Increment to invalidate any pending markers for the interrupted response
    
    # 4. Emit status update
    socketio.start_background_task(socketio.emit, 'status', {'message': "Response interrupted. Listening..."})

    print("[INTERRUPT] Response stopped by user.")


# Start workers upon server start
threading.Thread(target=tts_synth_worker, daemon=True).start()
threading.Thread(target=tts_play_worker, daemon=True).start()

# ============================== LLM ===============================
def gpt_stream_and_queue_tts(segment_text: str, pipeline_t0: float):
    """
    Single LLM pass per user segment.
    Streams tokens, builds clauses, enqueues them for TTS, and EMITS tokens to the frontend.
    """
    global conversation_context, RESPONSE_GEN_ID, rms_values

    with LLM_LOCK:
        STOP_SIGNAL.clear() # Clear the stop signal at the start of a new LLM call
        my_gen = RESPONSE_GEN_ID

        conversation_context.append({"role": "user", "content": segment_text})
        # cap context (keep system + last MAX_HISTORY pairs)
        if len(conversation_context) > (2 * MAX_HISTORY + 1):
            conversation_context[:] = [conversation_context[0]] + conversation_context[-(2 * MAX_HISTORY):]

        print(f"\n\n[USER] {segment_text}")
        
        full_response = ""
        try:
            resp = client_openai.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation_context,
                stream=True
            )

            t_llm_start = time.time()
            first_token_time = None

            clause_buf = ""
            sent_start_marker = False

            for chunk in resp:
                if STOP_SIGNAL.is_set(): # <-- CHECK FOR INTERRUPT SIGNAL
                    print("[LLM] Interrupted.")
                    # Remove the user message from context if response was aborted immediately
                    if conversation_context[-1]["role"] == "user":
                         conversation_context.pop()
                    return # ABORT LLM GENERATION

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    token = delta.content
                    full_response += token
                    clause_buf += token
                    
                    # EMIT TOKEN TO FRONTEND
                    socketio.emit('assistant_stream', {'token': token})

                    if first_token_time is None:
                        first_token_time = time.time()
                        latency_stats["llm_first_token_ms"] = int((first_token_time - t_llm_start) * 1000)

                    # Check for end of clause/sentence to send to TTS
                    if any(p in token for p in ".?!;:") and len(clause_buf.strip()) > 2:
                        if not sent_start_marker:
                            tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
                            sent_start_marker = True
                        tts_synth_q.put({"type": "text", "text": clause_buf.strip(), "gen_id": my_gen})
                        clause_buf = ""

            # Handle remaining text after stream ends
            if clause_buf.strip():
                if not sent_start_marker:
                    tts_synth_q.put({"type": "start", "pipeline_t0": pipeline_t0, "gen_id": my_gen})
                    sent_start_marker = True
                tts_synth_q.put({"type": "text", "text": clause_buf.strip(), "gen_id": my_gen})

            if sent_start_marker:
                tts_synth_q.put({"type": "end", "gen_id": my_gen})
            
            # Record LLM total time
            latency_stats["llm_total_ms"] = int((time.time() - t_llm_start) * 1000)

            # Update conversation history
            conversation_context.append({"role": "assistant", "content": full_response})
            
            print(f"[ASSISTANT] {full_response}")
            print("-" * 30)

            rms_values.clear()  # reset VAD after response completes

        except Exception as e:
            print(f"\n[LLM error] {e}")
            socketio.emit('error', {'message': f'LLM Generation Error: {e}'})
            
            # If an error occurred, ensure the context is clean
            if conversation_context[-1]["role"] == "user":
                 conversation_context.pop() # Remove the user prompt
            if conversation_context and conversation_context[-1]["role"] == "assistant":
                 conversation_context.pop() # Remove incomplete assistant response

            rms_values.clear()


# ============================== CORE LOOP ===============================
def live_transcribe_loop():
    """
    Mic -> VAD -> STT (tiny.en) -> LLM -> TTS.
    Runs in a dedicated thread.
    """
    global last_user_text, RESPONSE_GEN_ID
    
    # EMIT initial status to frontend
    socketio.emit('status', {'message': "Loading Whisper model..."})
    try:
        model = whisper.load_model(WHISPER_MODEL)
    except Exception as e:
        socketio.emit('error', {'message': f'Failed to load Whisper model: {e}'})
        print(f"Failed to load Whisper model: {e}")
        return

    socketio.emit('status', {'message': "Listening..."})
    print("Listening... (Ctrl+C to exit)")


    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                # Use socketio.sleep instead of time.sleep or queue.get(timeout)
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

                enough_audio = buffer.size >= int(MIN_CHUNK_SEC * SAMPLE_RATE)
                paused_long = last_speech_time and (time.time() - last_speech_time > PAUSE_TO_CUT_SEC)

                if enough_audio and paused_long:
                    stt_t0 = time.time()
                    pipeline_t0 = stt_t0

                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                    socketio.emit('status', {'message': "Transcribing audio..."})
                    
                    print(f"\n[STT] Transcribing {len(audio_chunk)/SAMPLE_RATE:.2f}s...")
                    result = model.transcribe(audio_chunk, **WHISPER_ARGS)
                    text = (result.get("text") or "").strip()
                    latency_stats["stt_latency_ms"] = int((time.time() - stt_t0) * 1000)

                    if text:
                        if not looks_useful_text(text):
                            print(f"[STT] Ignored short/low-content: {text!r}")
                            socketio.emit('status', {'message': "Listening..."})
                            continue
                        sim = difflib.SequenceMatcher(None, text.lower(), last_user_text.lower()).ratio()
                        if sim > 0.85 or text.lower() in last_user_text.lower():
                            print(f"[STT] Suppressed duplicate: {text!r}")
                            socketio.emit('status', {'message': "Listening..."})
                            continue
                        last_user_text = text

                        # EMIT USER TRANSCRIPT TO FRONTEND
                        socketio.emit('transcript_update', {
                            'text': text, 
                            't_stt': latency_stats["stt_latency_ms"]
                        })
                        
                        if not LLM_LOCK.locked():
                            RESPONSE_GEN_ID += 1
                            threading.Thread(
                                target=gpt_stream_and_queue_tts,
                                args=(text, pipeline_t0),
                                daemon=True
                            ).start()
                        else:
                            socketio.emit('status', {'message': "LLM busy, waiting for next turn..."})

    except Exception as e:
        print(f"[Core error] {e}")
        socketio.emit('error', {'message': f'Core Loop Error: {e}'})
    finally:
        socketio.emit('status', {'message': "Assistant stopped."})


# ============================== FLASK ROUTES & SOCKET.IO EVENTS ===============================

@app.route('/')
def index():
    """Renders the HTML file."""
    # The frontend HTML must be in a folder named 'templates'
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Initial connection handler."""
    print('Client connected')
    emit('status', {'message': 'Connected. Initializing audio stream...'})
    
    # Start the core loop in a new thread only once when the first client connects
    if not hasattr(socketio, 'assistant_thread'):
        socketio.assistant_thread = socketio.start_background_task(live_transcribe_loop)

@socketio.on('disconnect')
def handle_disconnect():
    """Disconnection handler."""
    print('Client disconnected')

@socketio.on('stop_response')
def handle_stop_response():
    """Handles the event from the frontend to stop the current AI response."""
    # Run in a separate thread to avoid blocking the Socket.IO event handler
    threading.Thread(target=interrupt_response, daemon=True).start()


# ============================== MAIN ===============================
if __name__ == "__main__":
    # Ensure the templates directory exists for Flask to find index.html
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("Starting AI Voice Assistant server...")
    # socketio.run hosts the Flask app and manages the threads
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)