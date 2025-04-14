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

import ssl
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
q = queue.Queue()

# Initialize chat history
chat_history = []

CONFIG_FILE = "A:\IRIS\V3\config.py"

try:
    with open(CONFIG_FILE, 'r') as file:
        config = eval(file.read())
        if 'MONGODB_URI' not in config:
            raise ValueError("MONGODB_URI not found in config.py")
except Exception as e:
    exit(1)

MODEL = config["LLM_MODEL"]
SIZE = config["WHISPER_SIZE"]
whisper_model = whisper.load_model(SIZE, device=DEVICE)  # Changed initialization

PERSONALITY = config["FRIDAY"]

def create_ssl_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Update these paths to your certificate and key files
    context.load_cert_chain(
        certfile='../UI/src/server/certs/localhost+2.pem', 
        keyfile='../UI/src/server/certs/localhost+2-key.pem'
    )
    return context

def initialize_audio():
    """Initialize audio device with proper error handling"""
    try:
        # List available devices
        devices = sd.query_devices()
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
            *messages
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
    return temp_filename

def transcribe_audio(filename):
    """Transcribe audio using Whisper"""
    # Load audio using whisper's built-in audio loading
    result = whisper_model.transcribe(
        filename,
        fp16=True,  # Set to True if using GPU and want faster processing
        language='en',  # You can specify language if know
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
    
    # Make sure to use HTTPS URL here
    url = "https://10.53.1.209:6969/upload-audio"
    temp_audio = open(output_path, 'rb')
    files = {'audio': ('audio.wav', temp_audio, 'audio/wav')}

    try:
        print("Sending audio to visualizer...")
        # Disable SSL verification if you're using self-signed certificates
        response = requests.post(url, files=files, verify=False)

        if response.status_code != 200:
            print(f"Error sending audio: {response.status_code} - {response.text}")
            RETURN_STATE = False
        
        print(f"Audio successfully sent to visualizer: {response.text}")
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
        pass   

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
        
        print("Friday's thinking...")
        response_start_time = time.time()
        
        assistant_response = get_ollama_response(messages)
        
        response_time = time.time() - response_start_time
        
        print(f"\nFriday: {assistant_response}")
        print(f"Response time: {response_time:.2f}s")

        TTS(assistant_response)

        chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": timestamp()
        })
        
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
        
        print("\nReady for next input! (Hold SPACE to speak)")
        
        # Return the audio URL for the client to play
        return jsonify({
            "success": True,
            "message": "Audio processed successfully",
            })
    
    return jsonify({"error": "Could not transcribe audio"}), 400

if __name__ == '__main__':
    ssl_context = create_ssl_context()
    app.run(host='0.0.0.0', port=2002, ssl_context=ssl_context)