import sounddevice as sd
import numpy as np
import keyboard
import time

DURATION = 1  # seconds
THRESHOLD = 0.00002
SAMPLERATE = 44100

def detect_sound_loop(threshold=THRESHOLD):
    print("🎧 Listening... Press 'q' or Ctrl + C to quit.")

    try:
        while True:
            audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float64')
            sd.wait()
            volume = np.linalg.norm(audio) / len(audio)

            print(f"🔍 Volume: {volume:.5f}")

            if volume > threshold:
                print("🔊 Loud sound detected! Exiting.")
                break

            if keyboard.is_pressed('q'):
                print("👋 Manual exit triggered.")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Ctrl + C pressed. Exiting.")

if __name__ == "__main__":
    detect_sound_loop()
