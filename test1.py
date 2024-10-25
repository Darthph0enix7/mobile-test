import time
import numpy as np
import openwakeword
from openwakeword.model import Model
import sounddevice as sd

# Configuration
THRESHOLD = 0.1  # Confidence threshold for wake word detection
INFERENCE_FRAMEWORK = 'onnx'
MODEL_PATHS = ['maii__croft__.onnx', 'Necks_us_.onnx', '_Jarvis_.onnx']

# Audio parameters for wake word detection
RATE = 16000
CHUNK_DURATION_MS = 30  # Duration of a chunk in milliseconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # Number of samples per chunk
COOLDOWN = 1  # Seconds to wait before detecting another wake word

def detect_wakeword():
    # Initialize wakeword detection model
    owwModel = Model(
        wakeword_models=MODEL_PATHS,
        inference_framework=INFERENCE_FRAMEWORK
    )

    last_notification_time = 0
    print("\n\nListening for wakewords...\n")
    
    def audio_callback(indata, frames, time, status):
        nonlocal last_notification_time
        if status:
            print(status, flush=True)
        mic_audio = indata[:, 0]

        # Amplify the audio by a factor of 2
        amplified_audio = mic_audio * 4

        # Feed to wakeword model
        prediction = owwModel.predict(amplified_audio)

        # Check for detected wakewords
        detected_models = [mdl for mdl in prediction.keys() if prediction[mdl] >= THRESHOLD]
        if detected_models and (time.time() - last_notification_time) >= COOLDOWN:
            last_notification_time = time.time()
            
            # Print the detected model
            detected_model = detected_models[0]
            print(f'Detected activation from \"{detected_model}\" model!')

    with sd.InputStream(samplerate=RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        print("Press Ctrl+C to stop the script")
        while True:
            time.sleep(0.1)

# Run the wakeword detection
detect_wakeword()