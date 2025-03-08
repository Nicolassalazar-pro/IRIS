import sounddevice as sd
import numpy as np
import keyboard
import queue
import json
import datetime
import pytz
import threading
import wave
import os
from faster_whisper import WhisperModel
import ollama
import time
import sys

print("\n=== Initializing Voice Chat System ===")

# Initialize Whisper model with CPU settings
print("📝 Loading Whisper model...")
whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    download_root="models",
    num_workers=2  # Increased for better CPU performance
)

# Initialize audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
q = queue.Queue()

# Initialize chat history
chat_history = []

def timestamp():
    """Get current Eastern time timestamp"""
    tz = pytz.timezone('US/Eastern')
    return datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S EST')

def get_ollama_response(messages):
    """Get response from Ollama"""
    try:
        response = ollama.chat(
            model='mistral',
            messages=messages,
            options={
                "mirostat": 2,
                "mirostat_tau": 0.8,
                "num_ctx": 2048,
                "num_thread": 6,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

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
    segments, info = whisper.transcribe(
        filename,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    return " ".join([segment.text for segment in segments])

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

def main():
    print("\n=== Voice Chat Ready ===")
    print("\nSystem Status:")
    print("🎤 Speech: Whisper small (CPU mode)")
    print("🤖 Chat: Mistral (GPU auto-detection)")
    
    print("\nControls:")
    print("- Press and HOLD SPACE to record speech")
    print("- Release SPACE to process and get response")
    print("- Press ESC to exit")
    print("\nReady to chat! 🎤")
    
    # Set up audio stream
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=2048,
        dtype=DTYPE,
        channels=CHANNELS,
        callback=audio_callback
    )

    # Load existing chat history
    try:
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            chat_history.extend(json.load(f))
    except FileNotFoundError:
        pass

    with stream:
        while True:
            if keyboard.is_pressed('space'):
                recording_start_time = time.time()
                while keyboard.is_pressed('space'):
                    elapsed_time = time.time() - recording_start_time
                    sys.stdout.write(f'\r🎙️ Recording... [{elapsed_time:.1f}s] [SPACE held] ')
                    sys.stdout.flush()
                    time.sleep(0.1)
                
                sys.stdout.write('\r' + ' ' * 50 + '\r')
                sys.stdout.flush()
                
                print("\n🔄 Processing speech...")
                text = process_audio()
                
                if text:
                    print(f"\n👤 You: {text}")
                    
                    chat_history.append({
                        "role": "user",
                        "content": text,
                        "timestamp": timestamp()
                    })
                    
                    messages = [
                        {"role": m["role"], "content": m["content"]} 
                        for m in chat_history[-5:]
                    ]
                    
                    print("🤖 Getting AI response...")
                    response_start_time = time.time()
                    assistant_response = get_ollama_response(messages)
                    response_time = time.time() - response_start_time
                    
                    print(f"\n🤖 Assistant: {assistant_response}")
                    print(f"⏱️ Response time: {response_time:.2f}s")
                    
                    chat_history.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": timestamp()
                    })
                    
                    with open('chat_history.json', 'w', encoding='utf-8') as f:
                        json.dump(chat_history, f, indent=2, ensure_ascii=False)
                    
                    print("\nReady for next input! (Hold SPACE to speak)")
            
            if keyboard.is_pressed('esc'):
                print("\n👋 Exiting...")
                break
            
            time.sleep(0.01)

if __name__ == "__main__":
    main()
