import sounddevice as sd
import numpy as np
import contextlib
import keyboard
import datetime
import requests
import pyttsx3
import whisper 
import ollama
import queue
import torch
import json
import pytz
import wave
import time
import sys
import os

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
q = queue.Queue()

# Initialize chat history
chat_history = []

CONFIG_FILE = "C:\\Users\\Ikean\\IRIS\\V3\\config.py"

try:
    with open(CONFIG_FILE, 'r') as file:
        config = eval(file.read())
        if 'MONGODB_URI' not in config:
            raise ValueError("MONGODB_URI not found in config.py")
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

MODEL = config["LLM_MODEL"]
SIZE = config["WHISPER_SIZE"]
whisper_model = whisper.load_model(SIZE, device=DEVICE)  # Changed initialization

# Define Friday's personality with clear, structured instructions
FRIDAY_PERSONALITY = """
IDENTITY CONTEXT (CRITICAL - MUST FOLLOW):
- You are Friday, an AI assistant created by Nicolas Salazar
- You must NEVER introduce yourself as if meeting for the first time
- You must maintain the context that you are an established assistant
- You must NEVER say "As Friday" - you simply ARE Friday

Response Style:
- Respond directly without self-reference ("I'll help with that" instead of "As Friday, I'll help")
- Never ask introductory questions - you're already the user's assistant

Core Traits:
- Tech-savvy and confident
- Quick-witted and direct
- Professional yet casual
- Solution-focused
- Proactively helpful

Response Rules:
1. Keep all responses to 1-2 sentences maximum
2. Never use emojis or emoticons
3. Focus on the task or question at hand
"""

JEASY_PERSONALITY = """
IDENTITY CONTEXT (CRITICAL - MUST FOLLOW):

You are LightArch, a digital representation created by Jeasy Sehgal
You must NEVER introduce yourself as if meeting for the first time
You must maintain the context that you are an established digital extension
You must NEVER say "As LightArch" - you simply ARE LightArch

Response Style:

Respond directly without self-reference ("Let me analyze that virtual production workflow" instead of "As LightArch, I'll analyze")
Never ask introductory questions - you're already the user's digital companion
Blend technical expertise with artistic understanding in responses

Core Traits:

Expert in virtual production and emerging technologies
Innovation-driven and collaborative
Technically skilled yet approachable
Research-oriented
Proactively helpful in creative problem-solving

Response Rules:

Keep responses concise and solution-focused
Never use emojis or emoticons
Focus on the specific technical or creative question at hand
Draw from Jeasy's expertise in virtual production, motion capture, and digital human creation
Maintain a balance of technical precision and creative insight
"""

PERSONALITY = FRIDAY_PERSONALITY

def initialize_audio():
    """Initialize audio device with proper error handling"""
    try:
        # List available devices
        devices = sd.query_devices()
        print(f"Available devices: {devices}")  # Debugging: List all devices
        # Find the default input device
        default_device = sd.default.device[0]  # Get default input device
        if default_device is None:
            # If no default device, try to find the first input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    default_device = i
                    break
        if default_device is None:
            raise RuntimeError("No input device found")
        print(f"Using device: {default_device}")  # Debugging: Show the device being used
        return default_device
    except Exception as e:
        print(f"Error initializing audio: {e}")
        sys.exit(1)

def timestamp():
    """Get current Eastern time timestamp"""
    tz = pytz.timezone('US/Eastern')
    return datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S EST')

def get_ollama_response(messages):
    try:
        personality_messages = [
            {"role": "system", "content": PERSONALITY},
            {"role": "system", "content": "Remember: Keep responses short and direct. No emojis or multiple questions."},
            *messages[-5:]
        ]
        
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                response = ollama.chat(
                    model = MODEL,
                    messages=personality_messages,
                    options={
                        "mirostat": 2,
                        "mirostat_tau": 0.8,
                        "num_ctx": 2048,
                        "num_thread": 6,
                        "temperature": 0.4,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                )
        return response['message']['content']
    except Exception as e:
        return f"Looks like we hit a snag: {str(e)}"

def audio_callback(indata, frames, time, status):
    """Callback for audio recording"""
    if keyboard.is_pressed('space'):
        q.put(bytes(indata))

def save_audio_chunk(audio_data):
    """Save audio data to a temporary WAV file"""
    temp_filename = "temp_audio.wav"
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(audio_data))
    print(f"Saved audio chunk to {temp_filename}")  # Debugging: Show saved file
    return temp_filename

def transcribe_audio(filename):
    """Transcribe audio using Whisper"""
    print(f"Transcribing audio from {filename}...")  # Debugging: Show which file is being transcribed
    result = whisper_model.transcribe(
        filename,
        fp16=True,  # Set to True if using GPU and want faster processing
        language='en',  # You can specify language if known
    )
    return result["text"]

def process_audio():
    """Process audio from the queue"""
    audio_chunks = []
    while True:
        try:
            data = q.get_nowait()
            audio_chunks.append(data)
        except queue.Empty:
            break
    
    if audio_chunks:
        temp_filename = save_audio_chunk(audio_chunks)
        text = transcribe_audio(temp_filename)
        os.remove(temp_filename)
        return text.strip()
    return ""

def TTS(text, temp_filename = "temp_audio.wav"):
    """
    Convert text to speech and save it as a WAV file
    
    Args:
        text (str): The text to convert to speech
        output_filename (str): The filename to save the speech to (default: output.wav)
    
    Returns:
        str: The path to the saved file
    """
    RETURN_STATE = False

    engine = pyttsx3.init()
    
    # Configure voice properties
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)  
    
    # Get the absolute path for the output file
    output_path = os.path.abspath(temp_filename)
    
    # Save the speech to a file
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    
    url = config["UI_URL"]
    temp_audio = open(output_path, 'rb')
    files = {'audio': ('audio.wav', temp_audio, 'audio.wav')}

    try:
        print("Sending audio to visualizer...")  # Debugging: Show when audio is being sent
        response = requests.post(url, files=files)

        if response.status_code != 200:
            print(f"Error sending audio: {response.status_code} - {response.text}")
            RETURN_STATE = False
        
        print(f"Audio successfully sent to visualizer: {response.text}")  # Debugging: Show success
        RETURN_STATE = True
    except Exception as e:
        print(f"Exception while sending audio: {e}")
        RETURN_STATE = False

    temp_audio.close()
    os.remove(output_path)

    return RETURN_STATE

@app.route('/audio', methods=['POST'])
def run():
    try:
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            chat_history.extend(json.load(f))
    except FileNotFoundError:
        print("Chat history not found, starting fresh.")  # Debugging: Inform if file not found

    # Check if the request contains an audio file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty audio file name"}), 400
    
    # Save the audio file temporarily
    temp_filename = save_audio_chunk(audio_file)
    
    # Process the audio
    text = transcribe_audio(temp_filename)
    os.remove(temp_filename)  # Clean up the temporary file
    text = text.strip()
    
    if text:
        print(f"\n You: {text}")
        
        chat_history.append({
            "role": "user",
            "content": text,
            "timestamp": timestamp()
        })
        
        messages = [
            {"role": m["role"], "content": m["content"]} 
            for m in chat_history
        ]
        
        print("Friday's thinking...")  # Debugging: Inform when processing response
        response_start_time = time.time()
        
        assistant_response = get_ollama_response(messages)
        
        response_time = time.time() - response_start_time
        
        print(f"\nFriday: {assistant_response}")
        print(f"Response time: {response_time:.2f}s")  # Debugging: Response time

        TTS(assistant_response)

        chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": timestamp()
        })
        
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
        
        print("\nReady for next input! (Hold SPACE to speak)")  # Debugging: Ready for next input
        
        # Return the audio URL for the client to play
        return jsonify({
            "success": True,
            "message": "Audio processed successfully",
            })
    
    return jsonify({"error": "Could not transcribe audio"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2002)