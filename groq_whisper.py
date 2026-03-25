"""
Free Wispr - Tap fn to start/stop recording.
Menubar icon changes state. Uses Groq Whisper API with Hugging Face fallback.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
from groq import Groq
from scipy.io.wavfile import write as wav_write

import objc
from AppKit import NSSound
from AppKit import (
    NSApplication, NSColor,
    NSMakeRect, NSBezierPath,
    NSApplicationActivationPolicyAccessory,
    NSStatusBar, NSVariableStatusItemLength, NSMenu, NSMenuItem,
    NSEvent, NSFlagsChangedMask, NSFunctionKeyMask,
    NSImage, NSSize, NSTimer, NSRunLoop,
)

SAMPLE_RATE = 16000
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

groq_client = None

# State
recording = False
processing = False
audio_frames = []
status_item = None
state_lock = threading.Lock()

FN_KEYCODE = 63


def create_mic_image(state="idle"):
    size = NSSize(18, 18)
    image = NSImage.alloc().initWithSize_(size)
    image.lockFocus()

    if state == "recording":
        NSColor.colorWithRed_green_blue_alpha_(0.9, 0.15, 0.15, 1.0).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(0, 0, 18, 18)).fill()
        NSColor.whiteColor().set()
    elif state == "processing":
        NSColor.colorWithRed_green_blue_alpha_(1.0, 0.6, 0.0, 1.0).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(0, 0, 18, 18)).fill()
        NSColor.whiteColor().set()
    else:
        NSColor.colorWithRed_green_blue_alpha_(0.4, 0.4, 0.4, 1.0).set()

    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(7, 6, 4, 8), 2, 2
    ).fill()

    arc = NSBezierPath.bezierPath()
    arc.setLineWidth_(1.2)
    arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
        (9, 9), 5, 0, 180, True
    )
    arc.stroke()

    line = NSBezierPath.bezierPath()
    line.setLineWidth_(1.2)
    line.moveToPoint_((9, 4))
    line.lineToPoint_((9, 2))
    line.moveToPoint_((6, 2))
    line.lineToPoint_((12, 2))
    line.stroke()

    image.unlockFocus()
    image.setTemplate_(state == "idle")
    return image


class ToggleHelper(objc.lookUpClass("NSObject")):
    def toggleRecording_(self, sender):
        toggle_recording()

toggle_helper = None

def create_menubar():
    global status_item, toggle_helper
    toggle_helper = ToggleHelper.alloc().init()
    status_bar = NSStatusBar.systemStatusBar()
    status_item = status_bar.statusItemWithLength_(NSVariableStatusItemLength)
    status_item.button().setImage_(create_mic_image("idle"))

    menu = NSMenu.alloc().init()
    toggle_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Toggle Recording", "toggleRecording:", "r"
    )
    toggle_item.setTarget_(toggle_helper)
    menu.addItem_(toggle_item)
    menu.addItem_(NSMenuItem.separatorItem())
    quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Quit Free Wispr", "terminate:", "q"
    )
    menu.addItem_(quit_item)
    status_item.setMenu_(menu)


def update_menubar_icon(state="idle"):
    status_item.button().performSelectorOnMainThread_withObject_waitUntilDone_(
        "setImage:", create_mic_image(state), False
    )


def notify(message):
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "Free Wispr"'
    ], capture_output=True)


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
                        "You are a speech-to-prompt assistant. The user dictated a message "
                        "via voice. Clean it up into a clear, well-structured prompt or message. "
                        "Fix filler words, repetition, and unclear phrasing. Keep the original "
                        "intent and meaning. Do NOT add anything the user didn't say. "
                        "Do NOT wrap in quotes. Output ONLY the cleaned text, nothing else."
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
    subprocess.run(["pbcopy"], input=text.encode(), check=True)
    time.sleep(0.05)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], capture_output=True)


def play_sound(name):
    sound = NSSound.soundNamed_(name)
    if sound:
        sound.setVolume_(0.15)
        sound.play()


def _log(msg):
    with open("/tmp/groq-whisper.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())


# Persistent audio stream — opened once at startup, never closed.
# This avoids CoreAudio Pa_OpenStream deadlocks with the main run loop.
persistent_stream = None


def init_audio_stream():
    """Open the persistent audio stream. Call once at startup from a background thread."""
    global persistent_stream
    try:
        persistent_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_audio_callback
        )
        persistent_stream.start()
        _log("Persistent audio stream opened")
    except Exception as e:
        _log(f"Audio stream init error: {e}")


def do_start():
    """Start recording. Must be called with state_lock held."""
    global recording, audio_frames
    if recording or processing:
        return
    if not persistent_stream or not persistent_stream.active:
        _log("ERROR: No active audio stream")
        return
    recording = True
    audio_frames = []
    update_menubar_icon("recording")
    play_sound("Tink")
    _log("Recording started")


def do_stop_and_process():
    """Stop recording and process. Runs in a background thread."""
    global recording, processing

    # Grab state and stop accepting new frames
    with state_lock:
        if not recording:
            return
        recording = False
        processing = True
        frames = list(audio_frames)
        audio_frames.clear()

    _log("Stopping...")

    play_sound("Pop")

    try:
        if not frames:
            _log("No frames")
            return

        audio = np.concatenate(frames, axis=0)
        del frames
        rms = np.sqrt(np.mean(audio ** 2))
        duration = len(audio) / SAMPLE_RATE
        _log(f"Stopped. RMS={rms:.6f} Duration={duration:.1f}s")
        if rms < 0.0005:
            _log("Too quiet")
            return

        update_menubar_icon("processing")
        audio_int16 = (audio * 32767).astype(np.int16)
        del audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_write(f, SAMPLE_RATE, audio_int16)
            tmp_path = f.name
        del audio_int16

        # Save backup before API call
        backup_dir = os.path.expanduser("~/.local/groq-whisper-app/backups")
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
                play_sound("Ping")
                # Remove backup on success
                try:
                    os.unlink(backup_path)
                except Exception:
                    pass
        except Exception as e:
            _log(f"ERROR: {e}")
            notify(f"Failed: {str(e)[:50]} (backup saved)")
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
        update_menubar_icon("idle")


def toggle_recording():
    """Toggle between recording and not recording."""
    with state_lock:
        is_rec = recording
        is_proc = processing

    if is_rec:
        def _stop_with_deadline():
            do_stop_and_process()
        t = threading.Thread(target=_stop_with_deadline, daemon=True)
        t.start()
        def _deadline():
            t.join(timeout=600)
            if t.is_alive():
                global processing
                _log("DEADLINE: 10min exceeded, force reset")
                notify("Transcription timed out")
                with state_lock:
                    processing = False
                update_menubar_icon("idle")
        threading.Thread(target=_deadline, daemon=True).start()
    elif not is_proc:
        with state_lock:
            do_start()


poll_fn_was_down = False
poll_fn_down_time = 0
poll_fn_had_other = False
last_toggle_time = 0
poll_count = 0


def poll_fn_key():
    """Called by NSTimer every 0.05s on the main thread."""
    global poll_fn_was_down, poll_fn_down_time, poll_fn_had_other, last_toggle_time, poll_count

    try:
        poll_count += 1

        flags = NSEvent.modifierFlags()
        fn_is_down = bool(flags & NSFunctionKeyMask)

        if fn_is_down and not poll_fn_was_down:
            poll_fn_down_time = time.time()
            poll_fn_had_other = False
        elif not fn_is_down and poll_fn_was_down:
            elapsed = time.time() - poll_fn_down_time
            since_last = time.time() - last_toggle_time
            other_mods = flags & 0xFFFF0000 & ~NSFunctionKeyMask
            if not poll_fn_had_other and not other_mods and elapsed < 0.5 and since_last > 0.5:
                last_toggle_time = time.time()
                toggle_recording()
        elif fn_is_down:
            other_mods = flags & 0xFFFF0000 & ~NSFunctionKeyMask
            if other_mods:
                poll_fn_had_other = True

        poll_fn_was_down = fn_is_down
    except Exception as e:
        _log(f"POLL ERROR: {e}")


if __name__ == "__main__":
    if not GROQ_API_KEY:
        notify("Set GROQ_API_KEY environment variable")
        sys.exit(1)

    groq_client = Groq(api_key=GROQ_API_KEY, timeout=30.0)

    if HF_API_KEY:
        _log("HF fallback ready")
    else:
        _log("WARNING: No HF_API_KEY, no fallback")

    # Open persistent audio stream BEFORE the run loop starts
    # (must happen before NSApplication.run() to avoid CoreAudio deadlock)
    init_audio_stream()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    # Disable App Nap — prevents macOS from suspending timers/monitors
    import Foundation
    activity = Foundation.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        0x00FFFFFF,
        "Listening for fn key"
    )

    create_menubar()

    # Use BOTH approaches for maximum reliability:

    # 1. NSTimer polling fn key state (immune to monitor revocation)
    class FnPoller(objc.lookUpClass("NSObject")):
        def poll_(self, timer):
            poll_fn_key()

    global _fn_poller, _fn_timer
    _fn_poller = FnPoller.alloc().init()
    _fn_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.05, _fn_poller, "poll:", None, True
    )
    NSRunLoop.currentRunLoop().addTimer_forMode_(_fn_timer, Foundation.NSRunLoopCommonModes)

    # 2. NSEvent monitors as backup
    evt_state = {"fn_down": False, "fn_time": 0, "fn_other": False}

    def handle_flags_changed(event):
        global last_toggle_time
        try:
            keycode = event.keyCode()
            if keycode == FN_KEYCODE:
                flags = event.modifierFlags()
                fn_is_down = bool(flags & NSFunctionKeyMask)
                if fn_is_down and not evt_state["fn_down"]:
                    evt_state["fn_time"] = time.time()
                    evt_state["fn_other"] = False
                elif not fn_is_down and evt_state["fn_down"]:
                    elapsed = time.time() - evt_state["fn_time"]
                    since_last = time.time() - last_toggle_time
                    if not evt_state["fn_other"] and elapsed < 0.5 and since_last > 0.5:
                        last_toggle_time = time.time()
                        toggle_recording()
                evt_state["fn_down"] = fn_is_down
            else:
                evt_state["fn_other"] = True
        except Exception as e:
            _log(f"EVENT ERROR: {e}")

    global _global_monitor, _local_monitor
    _global_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        NSFlagsChangedMask, handle_flags_changed
    )
    _local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        NSFlagsChangedMask, lambda e: (handle_flags_changed(e), e)[1]
    )

    _log("App started (polling + monitors + App Nap disabled)")

    try:
        app.run()
    except KeyboardInterrupt:
        pass
