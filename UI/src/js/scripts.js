import * as THREE from 'three';
import {EffectComposer} from 'three/examples/jsm/postprocessing/EffectComposer';
import {RenderPass} from 'three/examples/jsm/postprocessing/RenderPass';
import {UnrealBloomPass} from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import {OutputPass} from 'three/examples/jsm/postprocessing/OutputPass';
import ParticleSystem from './ParticleSystem.js';

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#1e1e1e");

const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

renderer.outputColorSpace = THREE.SRGBColorSpace;

const renderScene = new RenderPass(scene, camera);

const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight));
bloomPass.threshold = 0;
bloomPass.strength = 0; // No bloom for a clean white look without glow
bloomPass.radius = 0;   

const bloomComposer = new EffectComposer(renderer);
bloomComposer.addPass(renderScene);
bloomComposer.addPass(bloomPass);

const outputPass = new OutputPass();
bloomComposer.addPass(outputPass);

camera.position.set(0, -2, 14);
camera.lookAt(0, 0, 0);

// First mesh (white wireframe)
const uniforms1 = {
    u_time: {type: 'f', value: 0.0},
    u_frequency: {type: 'f', value: 0.0},
    u_red: {type: 'f', value: 1.0},    // Pure white color
    u_green: {type: 'f', value: 1.0},  
    u_blue: {type: 'f', value: 1.0}    
}

const mat1 = new THREE.ShaderMaterial({
    uniforms: uniforms1,
    vertexShader: document.getElementById('vertexshader').textContent,
    fragmentShader: document.getElementById('fragmentshader').textContent
});

// Settings for the first mesh with the look you want
const Scale1 = 4;
const geo1 = new THREE.IcosahedronGeometry(1, 5); // Lower detail count, smaller radius

// Create and add the first mesh
const mesh1 = new THREE.Mesh(geo1, mat1);
mesh1.scale.set(Scale1, Scale1, Scale1);
scene.add(mesh1);
mesh1.material.wireframe = true;

// Second mesh (solid with color transitions)
const uniforms2 = {
    u_time: {type: 'f', value: 0.0},
    u_frequency: {type: 'f', value: 0.0},
    u_red: {type: 'f', value: 0.2},    // Starting with blue color
    u_green: {type: 'f', value: 0.4},  
    u_blue: {type: 'f', value: 0.8}    
}

const mat2 = new THREE.ShaderMaterial({
    uniforms: uniforms2,
    vertexShader: document.getElementById('vertexshader').textContent,
    fragmentShader: document.getElementById('fragmentshader').textContent,
    transparent: true,    // Enable transparency
    opacity: 0.7          // Make it slightly transparent
});

// Settings for the second mesh (smaller solid mesh)
const Scale2 = 1;
const geo2 = new THREE.IcosahedronGeometry(1, 5); // Same geometry but different scale

// Create and add the second mesh
const mesh2 = new THREE.Mesh(geo2, mat2);
mesh2.scale.set(Scale2, Scale2, Scale2);
scene.add(mesh2);
mesh2.material.wireframe = false; // Solid faces

// Create a wrapper group for mesh2 and its edges
const mesh2Group = new THREE.Group();
scene.add(mesh2Group);

// Remove mesh2 from scene and add it to the group instead
scene.remove(mesh2);
mesh2Group.add(mesh2);

// We need a different approach for the edges
// Instead of using EdgesGeometry, we'll create a wireframe version
// of the exact same geometry with the same shader material
const edgesMat = new THREE.ShaderMaterial({
    uniforms: uniforms2, // Share the same uniforms object
    vertexShader: document.getElementById('vertexshader').textContent,
    fragmentShader: document.getElementById('fragmentshader').textContent
});

// Override the fragment shader to always be black
edgesMat.fragmentShader = `
    void main() {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Pure black
    }
`;

// Clone the same geometry and make it wireframe
const edgesGeometry = geo2.clone(); // Use the exact same geometry

// Create mesh with wireframe
const edges = new THREE.Mesh(edgesGeometry, edgesMat);
edges.material.wireframe = true; // Make it wireframe
edges.scale.set(1.001, 1.001, 1.001); // Slightly larger to avoid z-fighting
mesh2Group.add(edges); // Add edges to the same group

// Variable to control mesh2 rotation speed (positive for clockwise rotation)
const mesh2_speed = .8; // Change this value to adjust rotation speed

// Rotation settings for first mesh
const rotationSpeed1 = {
  x: 0.05,
  y: 0.03,
  z: 0.02
};

// Random starting rotation offsets
const rotationOffset1 = {
  x: Math.random() * Math.PI * 2,
  y: Math.random() * Math.PI * 2,
  z: Math.random() * Math.PI * 2
};

// Random starting rotation offset for mesh2
const rotationOffset2 = Math.random() * Math.PI * 2;

// Create and add our particle system
const particles = new ParticleSystem(1500, 3, scene);

// Make sure audio context is created properly
let audioContextInitialized = false;
let audioContext;
const initAudioContext = () => {
  if (audioContextInitialized) return audioContext;
  
  // Create audio context first
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioContext = new AudioContext();
  
  // Resume it (needed for Chrome and other browsers with autoplay policies)
  if (audioContext.state === 'suspended') {
    audioContext.resume();
  }
  
  audioContextInitialized = true;
  return audioContext;
};

// Initialize on first user interaction
window.addEventListener('click', initAudioContext, { once: true });
window.addEventListener('touchstart', initAudioContext, { once: true });

const listener = new THREE.AudioListener();
camera.add(listener);

const sound = new THREE.Audio(listener);
const analyser = new THREE.AudioAnalyser(sound, 32);

// Track if we're currently processing an audio file
let isProcessingAudio = false;
let currentAudioId = null;

// RECORDING FUNCTIONALITY
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingStartTime;
let recordingStatus = document.createElement('div');

// Setup recording status indicator
recordingStatus.style.position = 'fixed';
recordingStatus.style.bottom = '20px';
recordingStatus.style.left = '20px';
recordingStatus.style.padding = '10px 20px';
recordingStatus.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
recordingStatus.style.color = 'white';
recordingStatus.style.borderRadius = '4px';
recordingStatus.style.fontFamily = 'Arial, sans-serif';
recordingStatus.style.zIndex = '1000';
recordingStatus.style.display = 'none';
document.body.appendChild(recordingStatus);

// Setup media recorder
const setupMediaRecorder = async () => {
  try {
    // Request audio with specific constraints to match server expectations
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: { 
        channelCount: 1,          // Mono audio
        sampleRate: 16000,        // 16kHz sample rate to match the server
        echoCancellation: true,   // Improve audio quality
        noiseSuppression: true    // Improve audio quality
      } 
    });
    
    // WebM is more widely supported than WAV in MediaRecorder
    mediaRecorder = new MediaRecorder(stream, { 
      mimeType: 'audio/webm;codecs=opus',
      audioBitsPerSecond: 128000  // Set bitrate for better quality
    });
    
    // Event handler for when data is available - request frequent chunks
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    // Event handler for when recording stops
    mediaRecorder.onstop = async () => {
      console.log(`Recording stopped, processing ${audioChunks.length} chunks`);
      
      try {
        // Create blob from audio chunks
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        // Convert WebM to WAV using Audio API
        // This is crucial for compatibility with your Flask server
        const wavBlob = await webmToWav(audioBlob);
        
        // Send audio to server
        sendAudioToServer(wavBlob);
        
        // Reset audio chunks
        audioChunks = [];
        
        // Update UI
        updateRecordingStatus(false);
      } catch (err) {
        console.error('Error processing audio:', err);
        updateRecordingStatus(false);
      }
    };
    
    console.log('Media recorder setup successfully with mime type:', mediaRecorder.mimeType);
  } catch (err) {
    console.error('Error setting up media recorder:', err);
    alert('Could not access microphone. Please check permissions and try again.');
  }
};

// Convert WebM audio blob to WAV format
// This is crucial for compatibility with the Flask server
async function webmToWav(webmBlob) {
  return new Promise((resolve, reject) => {
    console.log("Converting WebM to WAV...");
    
    // Create an audio context
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000 // Match the expected sample rate
    });
    
    // Create a file reader to read the blob
    const fileReader = new FileReader();
    
    fileReader.onload = async function(event) {
      try {
        // Decode the audio data
        const audioData = await audioContext.decodeAudioData(event.target.result);
        console.log("Audio decoded successfully:", audioData);
        
        // Convert to WAV format
        const wavBuffer = audioBufferToWav(audioData);
        console.log("Converted to WAV format");
        
        // Create WAV blob
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        console.log("Created WAV blob:", wavBlob);
        
        resolve(wavBlob);
      } catch (err) {
        console.error("Error in decoding audio:", err);
        reject(err);
      }
    };
    
    fileReader.onerror = function(error) {
      console.error("FileReader error:", error);
      reject(error);
    };
    
    // Read the WebM blob as an array buffer
    fileReader.readAsArrayBuffer(webmBlob);
  });
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
  const numChannels = 1; // Mono
  const sampleRate = 16000; // 16kHz
  const bitsPerSample = 16; // 16 bits per sample
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  
  // Get the PCM data
  const samples = buffer.getChannelData(0);
  const numSamples = samples.length;
  
  // Create buffer for the WAV file
  const wavBuffer = new ArrayBuffer(44 + numSamples * bytesPerSample);
  const view = new DataView(wavBuffer);
  
  // Write WAV header
  // RIFF chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + numSamples * bytesPerSample, true);
  writeString(view, 8, 'WAVE');
  
  // fmt sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  
  // data sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, numSamples * bytesPerSample, true);
  
  // Write PCM samples
  const offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const pcmSample = sample < 0 ? sample * 32768 : sample * 32767;
    view.setInt16(offset + i * bytesPerSample, pcmSample, true);
  }
  
  return wavBuffer;
}

// Helper function to write strings to DataView
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// Start recording
const startRecording = () => {
  if (!mediaRecorder) {
    setupMediaRecorder().then(() => {
      if (mediaRecorder) startRecording();
    });
    return;
  }
  
  if (isRecording) return;
  
  try {
    // Start recording with frequent timeslices (100ms)
    // This provides smaller chunks for better processing
    audioChunks = [];
    mediaRecorder.start(100);
    isRecording = true;
    recordingStartTime = Date.now();
    
    // Update UI
    updateRecordingStatus(true);
    
    console.log('Recording started');
  } catch (err) {
    console.error('Error starting recording:', err);
  }
};

// Stop recording
const stopRecording = () => {
  if (!isRecording) return;
  
  try {
    // Stop recording
    mediaRecorder.stop();
    isRecording = false;
    
    // Update UI
    updateRecordingStatus(false);
    
    console.log('Recording stopped');
  } catch (err) {
    console.error('Error stopping recording:', err);
  }
};

// Update recording status and particles
const updateRecordingStatus = (recording) => {
  // Update status div
  recordingStatus.style.display = recording ? 'block' : 'none';
  
  const recordingDuration = recording ? ((Date.now() - recordingStartTime) / 1000).toFixed(1) : 0;
  recordingStatus.textContent = recording ? `Recording: ${recordingDuration}s` : '';
  
  // Update particles
  particles.setRecording(recording);
  
  // If recording, update the timer every 100ms
  if (recording) {
    setTimeout(() => {
      if (isRecording) {
        updateRecordingStatus(true);
      }
    }, 100);
  }
};

// Send audio to server
const sendAudioToServer = (wavBlob) => {
  console.log('Preparing to send WAV audio to server...', wavBlob);
  
  // Create form data
  const formData = new FormData();
  
  // Make sure we're sending as audio/wav with correct filename
  const file = new File([wavBlob], 'audio.wav', { type: 'audio/wav' });
  formData.append('audio', file);
  
  console.log('Sending audio to server...');
  
  // Add debug information to see what's being sent
  const fileInfo = {
    name: file.name,
    type: file.type,
    size: file.size
  };
  console.log('File being sent:', fileInfo);
  
  // Send to the Flask server - Use HTTPS here
  fetch('https://10.53.1.209:2002/audio', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    console.log('Server response status:', response.status);
    
    if (!response.ok) {
      return response.text().then(text => {
        console.error(`Server error response: ${text}`);
        throw new Error(`Server returned ${response.status}: ${text || response.statusText}`);
      });
    }
    
    return response.json();
  })
  .then(data => {
    console.log('Audio uploaded and processed successfully:', data);
  })
  .catch(error => {
    console.error('Error with audio upload:', error);
  });
};

function requestAudioDeletion() {
  // Only delete once
  if (!isProcessingAudio) return;
  
  console.log("Requesting audio deletion");
  
  fetch('https://10.53.1.209:6969/delete-audio', {
    method: 'POST'
  })
  .then(() => {
    console.log("Audio file deleted successfully");
    isProcessingAudio = false;
    currentAudioId = null;
  })
  .catch(err => {
    console.log('Error deleting audio:', err);
    isProcessingAudio = false;
    currentAudioId = null;
  });
}

// Function to load and play the audio
function loadAndPlayAudio(audioId) {
  // If we're already processing this file, skip
  if (isProcessingAudio && audioId === currentAudioId) {
    return;
  }
  
  isProcessingAudio = true;
  currentAudioId = audioId;
  console.log("Loading audio:", audioId);
  
  // Make sure audio context is initialized
  initAudioContext();
  
  const audioLoader = new THREE.AudioLoader();
  audioLoader.load(`https://10.53.1.209:6969/uploads/audio.wav?t=${audioId}`, 
    // Success callback
    function(buffer) {
      console.log("Audio loaded successfully");
      
      // If sound is already playing, stop it
      if (sound.isPlaying) {
        sound.stop();
      }
      
      // Get audio duration
      const duration = buffer.duration;
      console.log("Audio duration:", duration, "seconds");
      
      // Small delay before playing to ensure buffer is fully processed
      setTimeout(() => {
        sound.setBuffer(buffer);
        sound.setLoop(false); // Ensure no looping
        sound.play();
        console.log("Audio started playing");
        
        // Set both an event handler AND a timeout to ensure cleanup
        sound.onEnded = function() {
          console.log("onEnded event triggered");
          requestAudioDeletion();
        };
        
        // Backup timeout in case the onEnded event doesn't fire
        // Set it to slightly longer than the audio duration
        setTimeout(requestAudioDeletion, (duration * 1000) + 500);
      }, 100); // 100ms delay
    },
    // Progress callback
    function(xhr) {
      console.log("Audio loading progress: " + (xhr.loaded / xhr.total * 100) + "%");
    },
    // Error callback
    function(err) {
      console.error("Error loading audio:", err);
      isProcessingAudio = false;
      currentAudioId = null;
    }
  );
}

// Function to check for new audio
function checkForNewAudio() {
  // Skip checking if we're already processing an audio file or recording
  if (isProcessingAudio || isRecording) {
    return;
  }
  
  fetch('https://10.53.1.209:6969/uploads/audio.wav', { 
    method: 'HEAD',
    cache: 'no-store'
  })
    .then(response => {
      if (response.ok) {
        const audioId = Date.now(); // Generate a unique ID for this audio instance
        loadAndPlayAudio(audioId);
      }
    })
    .catch(error => {
      // File not found is normal and expected after deletion
    });
}


// Event listeners for keyboard and mouse
window.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && !isRecording) {
    event.preventDefault(); // Prevent scrolling
    startRecording();
  }
});

window.addEventListener('keyup', (event) => {
  if (event.code === 'Space' && isRecording) {
    stopRecording();
  }
});

renderer.domElement.addEventListener('mousedown', (event) => {
  if (event.button === 0 && !isRecording) { // Left mouse button
    startRecording();
  }
});

renderer.domElement.addEventListener('mouseup', (event) => {
  if (event.button === 0 && isRecording) { // Left mouse button
    stopRecording();
  }
});

// Handle case when mouse leaves window while recording
window.addEventListener('mouseleave', () => {
  if (isRecording) {
    stopRecording();
  }
});

// Initial cleanup when page loads
window.addEventListener('load', function() {
  console.log("Page loaded, initializing recording system");
  // Reset state on page load
  isProcessingAudio = false;
  currentAudioId = null;
  isRecording = false;
  
  // Initialize audio context and setup media recorder
  initAudioContext();
  setupMediaRecorder();
  
  // No need for cleanup since we're not using the old endpoint anymore
});

// Add cleanup handler when page unloads
window.addEventListener('beforeunload', function() {
  if (sound.isPlaying) {
    sound.stop();
    // Try to delete the file
    fetch('https://10.53.1.209:6969/delete-audio', {
      method: 'POST',
      // Use keepalive to ensure request completes even during page unload
      keepalive: true
    });
  }
  
  // Stop recording if it's in progress
  if (isRecording) {
    stopRecording();
  }
});

// Check for new audio every 100ms
setInterval(checkForNewAudio, 100);

const clock = new THREE.Clock();
function animate() {
  const time = clock.getElapsedTime();
  
  // Update uniforms for first mesh
  uniforms1.u_time.value = time;
  const frequency = analyser.getAverageFrequency();
  
  // This is the key adjustment - we multiply by 4 to match the original wave-to-size ratio
  uniforms1.u_frequency.value = frequency * ((Scale1 * .25)*.2);
  
  // Update uniforms for second mesh and its edges
  uniforms2.u_time.value = time;
  uniforms2.u_frequency.value = frequency * Scale2;
  
  // Create color transition for second mesh (cycling through blue tones)
  const bluePhase = (Math.sin(time * 0.3) + 1) / 2; // Value between 0-1
  // Mix between two different blue shades
  uniforms2.u_red.value = 0.1 + (bluePhase * 0.3);    // 0.1 to 0.4 (low red for blue)
  uniforms2.u_green.value = 0.2 + (bluePhase * 0.4);  // 0.2 to 0.6 (medium green for cyan-blue)
  uniforms2.u_blue.value = 0.6 + (bluePhase * 0.4);   // 0.6 to 1.0 (high blue)
  
  // Update the particle system with current time
  particles.update(time);
  particles.respondToAudio(frequency);
  
  // Rotate first mesh
  mesh1.rotation.x = rotationOffset1.x + (Math.sin(time * 0.4) * 0.2) + (time * rotationSpeed1.x);
  mesh1.rotation.y = rotationOffset1.y + (Math.sin(time * 0.3) * 0.3) + (time * rotationSpeed1.y);
  mesh1.rotation.z = rotationOffset1.z + (Math.sin(time * 0.7) * 0.1) + (time * rotationSpeed1.z);
  
  // Rotate second mesh group as a whole (both mesh and edges)
  mesh2Group.rotation.y = rotationOffset2 + (time * mesh2_speed);
  
  bloomComposer.render();
  requestAnimationFrame(animate);
}
animate();

window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    bloomComposer.setSize(window.innerWidth, window.innerHeight);
});