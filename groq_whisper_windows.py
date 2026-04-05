"""
Free Wispr (Windows) - Press Ctrl+Shift+Space to start/stop recording.
System tray icon changes state. Uses Groq Whisper API with HuggingFace fallback.
"""

import os
import sys
import threading
import time
import tempfile

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from groq import Groq
import pystray
from PIL import Image, ImageDraw
import keyboard
import pyperclip

SAMPLE_RATE = 16000
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
HOTKEY = os.environ.get("FREE_WISPR_HOTKEY", "ctrl+shift+space")

groq_client = None
recording = False
processing = False
audio_frames = []
state_lock = threading.Lock()
tray_icon = None


def create_tray_image(state="idle"):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if state == "recording":
        bg = (230, 38, 38, 255)
    elif state == "processing":
        bg = (255, 153, 0, 255)
    else:
        bg = (100, 100, 100, 255)

    draw.ellipse([0, 0, 63, 63], fill=bg)
    mic = (255, 255, 255, 255)
    draw.rounded_rectangle([26, 10, 38, 40], radius=6, fill=mic)
    draw.arc([16, 22, 48, 48], start=180, end=0, fill=mic, width=3)
    draw.line([32, 48, 32, 56], fill=mic, width=3)
    draw.line([24, 56, 40, 56], fill=mic, width=3)
    return img


def update_tray_icon(state="idle"):
    if tray_icon:
        tray_icon.icon = create_tray_image(state)


def notify(message):
    try:
        tray_icon.notify(message, "Free Wispr")
    except Exception:
        pass


LOG_PATH = os.path.join(os.path.expanduser("~"), "groq-whisper.log")

def _log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())


persistent_stream = None


def init_audio_stream():
    global persistent_stream
    try:
        persistent_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_audio_callback
        )
        persistent_stream.start()
        _log("Audio stream opened")
    except Exception as e:
        _log(f"Audio stream error: {e}")


def transcribe_groq(tmp_path):
    with open(tmp_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            file=(os.path.basename(tmp_path), f.read()),
            model="whisper-large-v3",
            language="en",
        )
    return result.text.strip()


def transcribe_huggingface(tmp_path):
    import requests
    with open(tmp_path, "rb") as f:
        audio_data = f.read()
    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "audio/wav",
        },
        data=audio_data,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict):
        return result.get("text", "").strip()
    return str(result).strip()


def clean_prompt(raw_text):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a transcription cleanup assistant. The user dictated text via voice. "
                        "Make only the minimum edits needed: remove obvious filler words (um, uh, like), "
                        "fix clear repetitions, and add punctuation. "
                        "Preserve the user's exact words, tone, and phrasing as much as possible. "
                        "Do NOT rephrase, rewrite, or restructure sentences. Do NOT wrap in quotes. "
                        "Output ONLY the lightly cleaned text, nothing else."
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        cleaned = response.choices[0].message.content.strip()
        return cleaned if cleaned else raw_text
    except Exception:
        return raw_text


def paste_text(text):
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.press_and_release("ctrl+v")


def do_start():
    global recording, audio_frames
    if recording or processing:
        return
    if not persistent_stream or not persistent_stream.active:
        _log("ERROR: No active audio stream")
        return
    recording = True
    audio_frames = []
    update_tray_icon("recording")
    _log("Recording started")


def do_stop_and_process():
    global recording, processing

    with state_lock:
        if not recording:
            return
        recording = False
        processing = True
        frames = list(audio_frames)
        audio_frames.clear()

    _log("Stopping...")

    try:
        if not frames:
            _log("No frames")
            return

        audio = np.concatenate(frames, axis=0)
        del frames
        rms = np.sqrt(np.mean(audio ** 2))
        duration = len(audio) / SAMPLE_RATE
        _log(f"RMS={rms:.6f} Duration={duration:.1f}s")
        if rms < 0.0005:
            _log("Too quiet")
            return

        update_tray_icon("processing")
        audio_int16 = (audio * 32767).astype(np.int16)
        del audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_write(f, SAMPLE_RATE, audio_int16)
            tmp_path = f.name
        del audio_int16

        backup_dir = os.path.join(os.path.expanduser("~"), ".free-wispr", "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.wav")
        try:
            import shutil
            shutil.copy2(tmp_path, backup_path)
        except Exception as e:
            _log(f"Backup failed: {e}")

        try:
            try:
                text = transcribe_groq(tmp_path)
                _log(f"Transcribed: '{text}'")
            except Exception as e:
                _log(f"Groq failed: {e}")
                notify("Groq down, using HF fallback")
                text = transcribe_huggingface(tmp_path)
                _log(f"HF fallback: '{text}'")

            if text:
                text = clean_prompt(text)
                paste_text(text)
                try:
                    os.unlink(backup_path)
                except Exception:
                    pass
        except Exception as e:
            _log(f"ERROR: {e}")
            notify(f"Failed: {str(e)[:50]}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        _log(f"FATAL: {e}")
    finally:
        with state_lock:
            processing = False
        update_tray_icon("idle")
        _log("Reset to idle")


last_toggle_time = 0


def toggle_recording():
    global last_toggle_time
    if time.time() - last_toggle_time < 0.5:
        return
    last_toggle_time = time.time()

    with state_lock:
        is_rec = recording
        is_proc = processing

    if is_rec:
        threading.Thread(target=do_stop_and_process, daemon=True).start()
    elif not is_proc:
        with state_lock:
            do_start()


def quit_app(icon):
    _log("Quitting")
    icon.stop()
    os._exit(0)


def main():
    global groq_client, tray_icon

    if not GROQ_API_KEY:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, "GROQ_API_KEY environment variable is not set.\n\nRun install.ps1 to set it up.",
            "Free Wispr", 0
        )
        sys.exit(1)

    groq_client = Groq(api_key=GROQ_API_KEY, timeout=30.0)

    init_audio_stream()

    keyboard.add_hotkey(HOTKEY, toggle_recording, suppress=False)
    _log(f"Hotkey: {HOTKEY}")

    menu = pystray.Menu(
        pystray.MenuItem(f"Toggle ({HOTKEY})", lambda icon, item: toggle_recording()),
        pystray.MenuItem("Quit", quit_app),
    )
    tray_icon = pystray.Icon(
        "Free Wispr",
        create_tray_image("idle"),
        "Free Wispr",
        menu,
    )

    _log("App started")
    tray_icon.run()


if __name__ == "__main__":
    main()
