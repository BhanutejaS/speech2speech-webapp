import streamlit as st
import openai
import speech_recognition as sr
import tempfile
import os
import time
import threading
from io import BytesIO
from datetime import datetime

# Optional imports with error handling
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    st.warning("PyAudio not available. Some audio features may be limited.")

try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Beyond Typing: Real-Time Conversational AI",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

class ConversationalAI:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        

    
    def speech_to_text(self, audio_data, stt_api_key=None, stt_provider="Google (Free)"):
        """Convert speech to text using various providers"""
        try:
            if stt_provider == "OpenAI Whisper" and stt_api_key:
                # Use OpenAI Whisper API
                return self.whisper_stt(audio_data, stt_api_key)
            elif stt_provider == "ElevenLabs STT" and stt_api_key:
                # Use ElevenLabs STT API
                return self.elevenlabs_stt(audio_data, stt_api_key)
            else:
                # Use free Google speech recognition
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
        except Exception as e:
            return f"STT error: {e}"
    
    def whisper_stt(self, audio_data, api_key):
        """Convert speech to text using OpenAI Whisper API"""
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Convert audio_data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # Convert AudioData to wav format
                wav_data = audio_data.get_wav_data()
                tmp_file.write(wav_data)
                tmp_file.flush()
                
                # Send to Whisper API
                with open(tmp_file.name, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
                return transcript.text
                
        except Exception as e:
            st.error(f"Whisper API error: {e}")
            # Fallback to Google STT
            return self.recognizer.recognize_google(audio_data)
    
    def elevenlabs_stt(self, audio_data, api_key):
        """Convert speech to text using ElevenLabs STT API"""
        try:
            import requests
            import time
            
            # Convert audio_data to a temporary file with better Windows handling
            tmp_file = None
            try:
                # Create temp file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                
                # Convert AudioData to wav format
                wav_data = audio_data.get_wav_data()
                tmp_file.write(wav_data)
                tmp_file.flush()
                tmp_file.close()  # Close file handle before reading
                
                # Small delay to ensure file is fully written
                time.sleep(0.1)
                
                # ElevenLabs STT API call
                url = "https://api.elevenlabs.io/v1/speech-to-text"
                headers = {
                    "xi-api-key": api_key
                }
                
                # Open file in binary mode for upload
                with open(tmp_file.name, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}  # Changed from "audio" to "file"
                    data = {"model_id": "scribe_v1"}  # Required model parameter
                    response = requests.post(url, headers=headers, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "No transcription returned")
                else:
                    st.error(f"ElevenLabs STT API error: {response.status_code} - {response.text}")
                    # Fallback to Google STT
                    return self.recognizer.recognize_google(audio_data)
                    
            finally:
                # Clean up temp file with retry logic for Windows
                if tmp_file and os.path.exists(tmp_file.name):
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            os.unlink(tmp_file.name)
                            break
                        except (PermissionError, OSError) as e:
                            if attempt < max_retries - 1:
                                time.sleep(0.2)  # Wait before retry
                            else:
                                st.warning(f"Could not delete temp file: {e}")
                
        except Exception as e:
            st.error(f"ElevenLabs STT error: {e}")
            # Fallback to Google STT
            return self.recognizer.recognize_google(audio_data)
    
    def generate_response(self, user_input, api_key, provider="Groq"):
        """Generate response using various LLM providers"""
        if provider == "OpenAI":
            return self.generate_openai_response(user_input, api_key)
        else:  # Groq
            return self.generate_groq_response(user_input, api_key)
    
    def generate_groq_response(self, user_input, api_key):
        """Generate response using Groq API"""
        try:
            import requests
            import json
            
            # Build conversation context
            messages = [
                {"role": "system", "content": "You are a helpful, conversational AI assistant. Keep responses natural, concise, and engaging for voice interaction."}
            ]
            
            # Add conversation history
            for msg in st.session_state.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            
            # Add current input
            messages.append({"role": "user", "content": user_input})
            
            # Groq API request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": messages,
                "model": "llama-3.1-8b-instant",  # Current fast Groq model
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": True
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=data, 
                stream=True
            )
            
            # Check if request was successful
            if response.status_code != 200:
                return f"API Error {response.status_code}: {response.text}"
            
            full_response = ""
            response_placeholder = st.empty()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        if line.startswith('data: [DONE]'):
                            break
                        try:
                            chunk_data = json.loads(line[6:])
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    response_placeholder.write(f"**AI:** {full_response}▌")
                        except json.JSONDecodeError as e:
                            st.warning(f"JSON decode error: {e}")
                            continue
                        except Exception as e:
                            st.warning(f"Chunk processing error: {e}")
                            continue
            
            response_placeholder.write(f"**AI:** {full_response}")
            
            # If no response was generated, try non-streaming
            if not full_response:
                st.info("Streaming failed, trying non-streaming response...")
                return self.generate_groq_non_streaming(user_input, api_key)
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_openai_response(self, user_input, api_key):
        """Generate response using OpenAI API"""
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Build conversation context
            messages = [
                {"role": "system", "content": "You are a helpful, conversational AI assistant. Keep responses natural, concise, and engaging for voice interaction."}
            ]
            
            # Add conversation history
            for msg in st.session_state.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            
            # Add current input
            messages.append({"role": "user", "content": user_input})
            
            # Generate streaming response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True,
                max_tokens=150,
                temperature=0.7
            )
            
            full_response = ""
            response_placeholder = st.empty()
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.write(f"**AI:** {full_response}▌")
            
            response_placeholder.write(f"**AI:** {full_response}")
            return full_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_groq_non_streaming(self, user_input, api_key):
        """Generate response using Groq API without streaming (fallback)"""
        try:
            import requests
            import json
            
            # Build conversation context
            messages = [
                {"role": "system", "content": "You are a helpful, conversational AI assistant. Keep responses natural, concise, and engaging for voice interaction."}
            ]
            
            # Add conversation history
            for msg in st.session_state.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            
            # Add current input
            messages.append({"role": "user", "content": user_input})
            
            # Groq API request (non-streaming)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": messages,
                "model": "llama-3.1-8b-instant",
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": False  # Non-streaming
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=data
            )
            
            if response.status_code != 200:
                return f"API Error {response.status_code}: {response.text}"
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "I apologize, but I couldn't generate a response. Please check your API key and try again."
                
        except Exception as e:
            return f"Non-streaming error: {str(e)}"
    
    def text_to_speech(self, text, tts_api_key=None, tts_provider="Silent (Text Only)"):
        """Convert text to speech using various TTS providers"""
        try:
            if tts_api_key == "edge_tts":
                # Use Edge TTS (free)
                self.edge_tts(text)
            elif tts_provider == "ElevenLabs TTS" and tts_api_key:
                # Use ElevenLabs TTS API
                self.elevenlabs_tts(text, tts_api_key)
            elif tts_provider == "OpenAI TTS" and tts_api_key:
                # Use OpenAI TTS API
                self.openai_tts(text, tts_api_key)
            else:
                # No TTS - silent mode
                pass
        except Exception as e:
            st.error(f"Text-to-speech error: {e}")
    
    def openai_tts(self, text, api_key):
        """Convert text to speech using OpenAI TTS API"""
        try:
            st.info("🔊 Generating speech with OpenAI TTS...")
            client = openai.OpenAI(api_key=api_key)
            
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                response.stream_to_file(tmp_file.name)
                st.success("✅ Speech generated, playing audio...")
                
                # Play the audio file
                try:
                    import pygame
                    pygame.mixer.quit()  # Ensure clean state
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    st.success("🎵 Audio playback completed!")
                    
                except Exception as audio_error:
                    st.error(f"Audio playback error: {audio_error}")
                    st.info("Audio file was generated but couldn't be played. Check your audio system.")
                
                # Clean up
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass  # Ignore cleanup errors
                
        except ImportError as ie:
            st.error(f"Import error: {ie}")
            st.warning("pygame not installed. Install with: pip install pygame")
            st.info("TTS requires pygame for audio playback. Install it to use OpenAI TTS.")
        except openai.AuthenticationError:
            st.error("❌ Invalid OpenAI API key for TTS")
        except openai.RateLimitError:
            st.error("❌ OpenAI API rate limit exceeded")
        except Exception as e:
            st.error(f"OpenAI TTS error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def edge_tts(self, text):
        """Convert text to speech using Microsoft Edge TTS (free)"""
        try:
            st.info("🔊 Generating speech with Edge TTS...")
            import edge_tts
            import asyncio
            
            # Use a popular Edge TTS voice
            voice = "en-US-AriaNeural"  # Natural female voice
            
            async def generate_speech():
                # Generate speech
                communicate = edge_tts.Communicate(text, voice)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    await communicate.save(tmp_file.name)
                    return tmp_file.name
            
            # Run the async function
            audio_file = asyncio.run(generate_speech())
            st.success("✅ Speech generated with Edge TTS, playing audio...")
            
            # Play the audio file
            try:
                import pygame
                pygame.mixer.quit()  # Ensure clean state
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                st.success("🎵 Edge TTS audio playback completed!")
                
            except Exception as audio_error:
                st.error(f"Audio playback error: {audio_error}")
                st.info("Audio file was generated but couldn't be played. Check your audio system.")
            
            # Clean up
            try:
                os.unlink(audio_file)
            except:
                pass  # Ignore cleanup errors
                
        except ImportError:
            st.error("edge-tts not installed. Install with: pip install edge-tts")
            st.info("Edge TTS provides free, high-quality text-to-speech.")
        except Exception as e:
            st.error(f"Edge TTS error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def elevenlabs_tts(self, text, api_key):
        """Convert text to speech using ElevenLabs TTS API"""
        try:
            st.info("🔊 Generating speech with ElevenLabs TTS...")
            import requests
            
            # ElevenLabs TTS API endpoint
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default Rachel voice
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            st.info(f"Sending request to: {url}")
            st.info(f"Text to synthesize: '{text}'")
            
            response = requests.post(url, headers=headers, json=data)
            
            st.info(f"Response status: {response.status_code}")
            st.info(f"Response content length: {len(response.content)} bytes")
            
            if response.status_code == 200:
                if len(response.content) > 0:
                    # Save to temporary file and play
                    tmp_file = None
                    try:
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tmp_file.write(response.content)
                        tmp_file.flush()
                        tmp_file.close()  # Close file handle
                        
                        st.success(f"✅ Speech generated with ElevenLabs ({len(response.content)} bytes), playing audio...")
                        
                        # Play the audio file
                        try:
                            import pygame
                            pygame.mixer.quit()  # Ensure clean state
                            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                            pygame.mixer.music.load(tmp_file.name)
                            pygame.mixer.music.play()
                            
                            # Wait for playback to finish
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                            
                            st.success("🎵 ElevenLabs audio playback completed!")
                            
                        except Exception as audio_error:
                            st.error(f"Audio playback error: {audio_error}")
                            st.info("Audio file was generated but couldn't be played. Check your audio system.")
                            st.info(f"Temp file path: {tmp_file.name}")
                        
                    finally:
                        # Clean up temp file with retry logic
                        if tmp_file and os.path.exists(tmp_file.name):
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    os.unlink(tmp_file.name)
                                    break
                                except (PermissionError, OSError) as e:
                                    if attempt < max_retries - 1:
                                        time.sleep(0.2)
                                    else:
                                        st.warning(f"Could not delete temp file: {e}")
                else:
                    st.error("ElevenLabs returned empty audio content")
            else:
                st.error(f"ElevenLabs TTS API error: {response.status_code} - {response.text}")
                
        except ImportError:
            st.error("requests not installed. Install with: pip install requests")
        except Exception as e:
            st.error(f"ElevenLabs TTS error: {e}")
            import traceback
            st.code(traceback.format_exc())

def record_audio():
    """Record audio from microphone"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.info("Listening... Speak now!")
            
            # Record audio with timeout
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return audio_data
            
    except sr.WaitTimeoutError:
        st.warning("No speech detected within timeout period")
        return None
    except Exception as e:
        st.error(f"Recording error: {e}")
        return None

def main():
    st.title("Beyond Typing: Real-Time Conversational AI")
    st.markdown("A low-latency, web-based conversational platform for real-time voice interactions")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Configuration section
        st.subheader("API Configuration")
        
        # STT Configuration
        st.write("**Speech-to-Text (STT)**")
        stt_provider = st.selectbox("STT Provider", 
                                   ["Google (Free)", "OpenAI Whisper", "ElevenLabs STT"],
                                   help="Choose your speech-to-text provider")
        
        if stt_provider == "OpenAI Whisper":
            stt_api_key = st.text_input("STT API Key (OpenAI)", type="password", 
                                       help="Enter your OpenAI API key for Whisper STT")
        elif stt_provider == "ElevenLabs STT":
            stt_api_key = st.text_input("STT API Key (ElevenLabs)", type="password", 
                                       help="Enter your ElevenLabs API key for STT")
        else:
            stt_api_key = ""
        
        # LLM Configuration
        st.write("**Language Model (LLM)**")
        llm_provider = st.selectbox("LLM Provider", 
                                   ["Groq", "OpenAI"],
                                   help="Choose your language model provider")
        
        if llm_provider == "Groq":
            llm_api_key = st.text_input("LLM API Key (Groq)", type="password", 
                                       help="Enter your Groq API key for fast inference")
        else:  # OpenAI
            llm_api_key = st.text_input("LLM API Key (OpenAI)", type="password", 
                                       help="Enter your OpenAI API key for GPT models")
        
        # TTS Configuration
        st.write("**Text-to-Speech (TTS)**")
        tts_provider = st.selectbox("TTS Provider", 
                                   ["Silent (Text Only)", "Edge TTS (Free)", "OpenAI TTS", "ElevenLabs TTS"],
                                   help="Choose your text-to-speech provider")
        
        if tts_provider == "OpenAI TTS":
            tts_api_key = st.text_input("TTS API Key (OpenAI)", type="password", 
                                       help="Enter your OpenAI API key for natural TTS")
        elif tts_provider == "ElevenLabs TTS":
            tts_api_key = st.text_input("TTS API Key (ElevenLabs)", type="password", 
                                       help="Enter your ElevenLabs API key for high-quality TTS")
        elif tts_provider == "Edge TTS (Free)":
            tts_api_key = "edge_tts"  # Use special marker for Edge TTS
        else:
            tts_api_key = ""
        

        
        # Performance metrics
        st.subheader("Performance Metrics")
        if 'last_response_time' in st.session_state:
            st.metric("Last Response Time", f"{st.session_state.last_response_time:.2f}s")
        if 'total_exchanges' in st.session_state:
            st.metric("Total Exchanges", st.session_state.total_exchanges)
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Initialize AI
    ai = ConversationalAI()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation")
        
        # Display conversation history
        conversation_container = st.container()
        with conversation_container:
            for i, exchange in enumerate(st.session_state.conversation_history):
                st.write(f"**You:** {exchange['user']}")
                st.write(f"**AI:** {exchange['assistant']}")
                st.write(f"*{exchange['timestamp']}*")
                st.divider()
    
    with col2:
        st.subheader("Voice Input")
        
        # Push-to-talk button
        if st.button("🎙️ Push to Talk", key="record_btn", help="Click and speak"):
            if not llm_api_key:
                st.error(f"Please enter your {llm_provider} API key in the sidebar")
            else:
                start_time = time.time()
                
                # Record audio
                with st.spinner("Recording..."):
                    audio_data = record_audio()
                
                if audio_data:
                    # Speech to text
                    with st.spinner("Converting speech to text..."):
                        user_input = ai.speech_to_text(audio_data, stt_api_key, stt_provider)
                    
                    if user_input and user_input != "Could not understand audio":
                        st.success(f"You said: {user_input}")
                        
                        # Generate AI response
                        with st.spinner("Generating response..."):
                            ai_response = ai.generate_response(user_input, llm_api_key, llm_provider)
                        
                        # Calculate response time
                        response_time = time.time() - start_time
                        st.session_state.last_response_time = response_time
                        
                        # Store conversation
                        exchange = {
                            "user": user_input,
                            "assistant": ai_response,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "response_time": response_time
                        }
                        st.session_state.conversation_history.append(exchange)
                        
                        # Update metrics
                        if 'total_exchanges' not in st.session_state:
                            st.session_state.total_exchanges = 0
                        st.session_state.total_exchanges += 1
                        
                        # Text to speech (auto-enabled when TTS is selected)
                        if tts_api_key:
                            with st.spinner("Converting response to speech..."):
                                threading.Thread(target=ai.text_to_speech, args=(ai_response, tts_api_key, tts_provider)).start()
                        
                        st.rerun()
                    else:
                        st.error("Could not understand the audio. Please try again.")
        
        # Alternative text input
        st.subheader("Text Input")
        text_input = st.text_input("Type your message:", key="text_input")
        
        if st.button("Send Text") and text_input:
            if not llm_api_key:
                st.error(f"Please enter your {llm_provider} API key in the sidebar")
            else:
                start_time = time.time()
                
                # Generate AI response
                with st.spinner("Generating response..."):
                    ai_response = ai.generate_response(text_input, llm_api_key, llm_provider)
                
                # Calculate response time
                response_time = time.time() - start_time
                st.session_state.last_response_time = response_time
                
                # Store conversation
                exchange = {
                    "user": text_input,
                    "assistant": ai_response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "response_time": response_time
                }
                st.session_state.conversation_history.append(exchange)
                
                # Update metrics
                if 'total_exchanges' not in st.session_state:
                    st.session_state.total_exchanges = 0
                st.session_state.total_exchanges += 1
                
                # Text to speech (auto-enabled when TTS is selected)
                if tts_api_key:
                    with st.spinner("Converting response to speech..."):
                        threading.Thread(target=ai.text_to_speech, args=(ai_response, tts_api_key, tts_provider)).start()
                
                st.rerun()
    
    # Performance information
    with st.expander("System Information"):
        st.write("**Features:**")
        st.write("- Real-time speech-to-text conversion")
        st.write("- Streaming LLM responses")
        st.write("- Natural text-to-speech output")
        st.write("- Low-latency conversation flow")
        st.write("- Performance metrics tracking")
        
        st.write("**Current Configuration:**")
        st.write(f"- Speech Recognition: {stt_provider}")
        if llm_provider == "Groq":
            model_name = "Llama-3.1-8b-instant"
        else:
            model_name = "GPT-3.5-turbo"
        st.write(f"- Language Model: {f'{llm_provider} {model_name}' if llm_api_key else 'Not configured'}")
        st.write(f"- Text-to-Speech: {tts_provider}")
        
        st.write("**API Key Status:**")
        stt_status = "✅ Configured" if stt_api_key else "⚠️ Using free option"
        st.write(f"- STT ({stt_provider}): {stt_status}")
        
        llm_status = "✅ Configured" if llm_api_key else "❌ Required for AI responses"
        st.write(f"- LLM ({llm_provider}): {llm_status}")
        
        if tts_provider == "Edge TTS (Free)":
            tts_status = "✅ Free Edge TTS"
        elif tts_provider == "ElevenLabs TTS":
            tts_status = "✅ Configured" if tts_api_key else "❌ API key required"
        elif tts_provider == "OpenAI TTS":
            tts_status = "✅ Configured" if tts_api_key else "❌ API key required"
        else:
            tts_status = "⚠️ Silent mode"
        st.write(f"- TTS ({tts_provider}): {tts_status}")
        
        if llm_api_key:
            if llm_provider == "Groq":
                st.success("💡 **Groq** provides very fast inference with generous free tiers - perfect for low-latency conversations!")
            else:
                st.info("💡 **OpenAI** provides high-quality responses with excellent language understanding.")
        
        if tts_provider == "Edge TTS (Free)":
            st.success("💡 **Edge TTS** provides free, high-quality voices from Microsoft - no API key required!")
        elif tts_provider == "ElevenLabs TTS":
            st.success("💡 **ElevenLabs** provides premium voice quality with emotional range and customization!")

if __name__ == "__main__":
    main()
