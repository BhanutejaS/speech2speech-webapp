# Speech2Speech Webapp

This project implements a **speech-to-speech pipeline** with live transcription, LLM integration, and TTS output.

---

## Python Version

- **Python 3.10.18** is required.

## Libraries

The Python dependencies are listed in `requirements.txt`:
- numpy  
- sounddevice  
- openai-whisper  
- openai  
- python-dotenv  
- kokoro  
- flask  

---

## System Dependencies

- **PortAudio** → required by `sounddevice` for audio input/output  
- **eSpeak NG** → required by **Kokoro TTS** for phonemization  

---

## Installing System Dependencies

### macOS (Homebrew)

```bash
# Install PortAudio (for sounddevice)
brew install portaudio

# Install eSpeak NG (for Kokoro TTS phonemization)
brew install espeak-ng
```
### Windows
# Download and install PortAudio from http://www.portaudio.com/download.html
# Download and install eSpeak NG from https://github.com/espeak-ng/espeak-ng/releases
---

##  Environment Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv speech2speech_venv
   source speech2speech_venv/bin/activate   # macOS/Linux
   speech2speech_venv\Scripts\activate      # Windows
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Set the eSpeak library path for Kokoro TTS:

   **Example:**
   
   **macOS (Homebrew):**
   ```python
   os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"
   ```
   
   **Windows:**
   ```python
   os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
   ```

---

## Run the App

1. Navigate to your project folder:
   ```bash
   cd /Users/bhanu/Documents/capstone/speech2speech-webapp/
   ```

2. Run the main application:
   ```bash
   python app1.py
   ```

3. Open the web app in your browser:
   ```
   http://127.0.0.1:5000
   ```
