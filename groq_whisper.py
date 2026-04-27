"""
Groq Whisper - Tap fn to start/stop recording.
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
    NSApplication, NSWindow, NSView, NSColor, NSFont,
    NSMakeRect, NSMakePoint, NSScreen, NSBezierPath,
    NSWindowStyleMaskBorderless, NSBackingStoreBuffered,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable, NSWindowStyleMaskFullSizeContentView,
    NSFloatingWindowLevel,
    NSApplicationActivationPolicyAccessory,
    NSStatusBar, NSVariableStatusItemLength, NSMenu, NSMenuItem,
    NSEvent, NSFlagsChangedMask, NSFunctionKeyMask,
    NSKeyDownMask, NSKeyUpMask,
    NSImage, NSSize, NSTimer, NSRunLoop,
    NSScrollView, NSTextView, NSButton, NSTextField, NSStackView,
    NSVisualEffectView,
)
import json

SAMPLE_RATE = 16000
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

groq_client = None

# State
recording = False
processing = False
audio_frames = []
stream = None
status_item = None
state_lock = threading.Lock()

FN_KEYCODE = 63


def create_mic_image(state="idle"):
    from AppKit import NSImageSymbolConfiguration

    if state == "recording":
        symbol = "mic.fill"
        color = NSColor.colorWithRed_green_blue_alpha_(0.9, 0.15, 0.15, 1.0)
    elif state == "processing":
        symbol = "mic.fill"
        color = NSColor.colorWithRed_green_blue_alpha_(1.0, 0.6, 0.0, 1.0)
    else:
        symbol = "mic"
        color = None

    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, "Free Wispr")
    if img is None:
        return None

    if color is not None:
        config = NSImageSymbolConfiguration.configurationWithPaletteColors_([color])
        img = img.imageWithSymbolConfiguration_(config)
    else:
        img.setTemplate_(True)

    return img


class ToggleHelper(objc.lookUpClass("NSObject")):
    def toggleRecording_(self, sender):
        toggle_recording()

    def updateIcon_(self, state):
        img = create_mic_image(state)
        status_item.button().setImage_(img)

toggle_helper = None

# Clipboard history — last 10 transcriptions, persisted to disk
transcription_history = []
MAX_HISTORY = 10
HISTORY_PATH = os.path.expanduser("~/.local/groq-whisper-app/history.json")


def load_history():
    global transcription_history
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                transcription_history = [tuple(x) for x in json.load(f)]
    except Exception as e:
        _log(f"History load error: {e}")
        transcription_history = []


def save_history():
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(transcription_history, f)
    except Exception as e:
        _log(f"History save error: {e}")


class CopyHelper(objc.lookUpClass("NSObject")):
    def initWithText_(self, text):
        self = objc.super(CopyHelper, self).init()
        if self is not None:
            self._text = text
        return self

    def copyText_(self, sender):
        subprocess.run(["pbcopy"], input=self._text.encode(), check=True)
        # Visual feedback — change button title briefly
        try:
            sender.setTitle_("Copied!")
            def reset():
                sender.setTitle_("Copy")
            from Foundation import NSOperationQueue
            import threading as _t
            _t.Timer(1.0, lambda: NSOperationQueue.mainQueue().addOperationWithBlock_(reset)).start()
        except Exception:
            pass

copy_helpers = []  # keep strong references to prevent GC


def rebuild_menu():
    """Rebuild the status item menu with current history. Call on main thread."""
    menu = NSMenu.alloc().init()

    toggle_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Toggle Recording", "toggleRecording:", "r"
    )
    toggle_item.setTarget_(toggle_helper)
    menu.addItem_(toggle_item)

    if transcription_history:
        menu.addItem_(NSMenuItem.separatorItem())
        header = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Recent Dictations", "", ""
        )
        header.setEnabled_(False)
        menu.addItem_(header)

        copy_helpers.clear()
        for ts, text in transcription_history:
            label = text if len(text) <= 60 else text[:57] + "…"
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                f"{ts}  {label}", "copyText:", ""
            )
            helper = CopyHelper.alloc().initWithText_(text)
            copy_helpers.append(helper)
            item.setTarget_(helper)
            menu.addItem_(item)

    menu.addItem_(NSMenuItem.separatorItem())
    quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Quit Free Wispr", "terminate:", "q"
    )
    menu.addItem_(quit_item)
    status_item.setMenu_(menu)


def add_to_history(text):
    ts = time.strftime("%Y-%m-%d %H:%M")
    transcription_history.insert(0, (ts, text))
    if len(transcription_history) > MAX_HISTORY:
        transcription_history.pop()
    save_history()
    from Foundation import NSOperationQueue
    NSOperationQueue.mainQueue().addOperationWithBlock_(rebuild_menu)


history_window = None


def show_history_picker():
    """Show a window with cards for each recent dictation, each with a Copy button."""
    from Foundation import NSOperationQueue

    def _show():
        global history_window

        if not transcription_history:
            subprocess.run([
                "osascript", "-e",
                'display notification "No recent dictations yet" with title "Free Wispr"'
            ], capture_output=True)
            return

        WIDTH = 580
        CARD_GAP = 10
        OUTER_PAD = 18
        TITLE_BAR = 38
        TS_TOP_PAD = 14   # space above timestamp
        TS_HEIGHT = 14
        TS_TEXT_GAP = 8   # gap between ts and text
        TEXT_BOTTOM_PAD = 16
        BTN_WIDTH = 76
        BTN_GAP = 14      # gap between text and copy button
        TEXT_WIDTH = WIDTH - 2 * OUTER_PAD - 2 * 18 - BTN_WIDTH - BTN_GAP  # 18 = inner card pad

        # Pre-measure card heights using a sizing NSTextField
        sizer = NSTextField.alloc().init()
        sizer.setEditable_(False)
        sizer.setBordered_(False)
        sizer.setBezeled_(False)
        sizer.setDrawsBackground_(False)
        sizer.setFont_(NSFont.systemFontOfSize_(13))
        sizer.cell().setWraps_(True)
        sizer.cell().setTruncatesLastVisibleLine_(True)

        cards = []
        line_height = 17
        max_lines = 6
        for ts, text in transcription_history:
            sizer.setStringValue_(text)
            ideal = sizer.cell().cellSizeForBounds_(
                NSMakeRect(0, 0, TEXT_WIDTH, max_lines * line_height + 4)
            )
            text_h = min(max_lines * line_height, max(line_height, int(ideal.height) + 2))
            card_h = TS_TOP_PAD + TS_HEIGHT + TS_TEXT_GAP + text_h + TEXT_BOTTOM_PAD
            cards.append((ts, text, card_h, text_h))

        n = len(cards)
        content_h = sum(c[2] for c in cards) + (n - 1) * CARD_GAP + OUTER_PAD * 2
        win_h = min(720, content_h + TITLE_BAR)

        if history_window is None:
            screen = NSScreen.mainScreen().frame()
            x = (screen.size.width - WIDTH) / 2
            y = (screen.size.height - win_h) / 2
            style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                     | NSWindowStyleMaskResizable
                     | NSWindowStyleMaskFullSizeContentView)
            history_window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, WIDTH, win_h),
                style, NSBackingStoreBuffered, False
            )
            history_window.setTitle_("Free Wispr")
            history_window.setReleasedWhenClosed_(False)
            history_window.setTitlebarAppearsTransparent_(True)
            history_window.setMovableByWindowBackground_(True)
            history_window.setBackgroundColor_(NSColor.clearColor())

        # Frosted-glass background
        bg = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIDTH, win_h)
        )
        bg.setMaterial_(7)  # NSVisualEffectMaterialHUDWindow — strong frosted feel
        bg.setBlendingMode_(0)  # BehindWindow
        bg.setState_(1)  # Active

        scroll_frame = NSMakeRect(0, 0, WIDTH, win_h - TITLE_BAR)
        scroll = NSScrollView.alloc().initWithFrame_(scroll_frame)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(0)
        scroll.setDrawsBackground_(False)

        doc = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, content_h))

        copy_helpers.clear()

        # Lay out cards top-down (most recent at top)
        running_y = content_h - OUTER_PAD
        for i, (ts, text, card_h, text_h) in enumerate(cards):
            running_y -= card_h
            y_pos = running_y

            card = NSView.alloc().initWithFrame_(
                NSMakeRect(OUTER_PAD, y_pos, WIDTH - 2 * OUTER_PAD, card_h)
            )
            card.setWantsLayer_(True)
            # Subtle frosted-white card
            card.layer().setBackgroundColor_(
                NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.7).CGColor()
            )
            card.layer().setCornerRadius_(14.0)
            card.layer().setBorderWidth_(0.5)
            card.layer().setBorderColor_(
                NSColor.colorWithRed_green_blue_alpha_(0, 0, 0, 0.06).CGColor()
            )
            card.layer().setShadowOpacity_(0.06)
            card.layer().setShadowRadius_(8.0)
            card.layer().setShadowOffset_(NSSize(0, -2))

            # Timestamp
            ts_label = NSTextField.alloc().initWithFrame_(
                NSMakeRect(18, card_h - TS_TOP_PAD - TS_HEIGHT, 240, TS_HEIGHT)
            )
            ts_label.setStringValue_(ts)
            ts_label.setEditable_(False)
            ts_label.setBordered_(False)
            ts_label.setBezeled_(False)
            ts_label.setDrawsBackground_(False)
            ts_label.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(10.5, 0))
            ts_label.setTextColor_(NSColor.tertiaryLabelColor())

            # Text body
            text_y = TEXT_BOTTOM_PAD
            text_label = NSTextField.alloc().initWithFrame_(
                NSMakeRect(18, text_y, TEXT_WIDTH, text_h)
            )
            text_label.setStringValue_(text)
            text_label.setEditable_(False)
            text_label.setSelectable_(True)
            text_label.setBordered_(False)
            text_label.setBezeled_(False)
            text_label.setDrawsBackground_(False)
            text_label.setFont_(NSFont.systemFontOfSize_(13))
            text_label.setTextColor_(NSColor.labelColor())
            text_label.setLineBreakMode_(4)
            text_label.cell().setWraps_(True)
            text_label.cell().setTruncatesLastVisibleLine_(True)

            # Copy button — modern, accent-coloured
            btn_y = (card_h - 26) / 2
            btn = NSButton.alloc().initWithFrame_(
                NSMakeRect(WIDTH - 2 * OUTER_PAD - 18 - BTN_WIDTH, btn_y, BTN_WIDTH, 26)
            )
            btn.setTitle_("Copy")
            btn.setBezelStyle_(15)  # NSBezelStyleInline — modern pill
            try:
                btn.setHasDestructiveAction_(False)
            except Exception:
                pass
            btn.setControlSize_(0)
            btn.setFont_(NSFont.systemFontOfSize_weight_(12, 0.3))
            helper = CopyHelper.alloc().initWithText_(text)
            copy_helpers.append(helper)
            btn.setTarget_(helper)
            btn.setAction_("copyText:")

            card.addSubview_(ts_label)
            card.addSubview_(text_label)
            card.addSubview_(btn)
            doc.addSubview_(card)

            running_y -= CARD_GAP

        scroll.setDocumentView_(doc)

        # Scroll to top so most recent is visible
        clip_h = scroll.contentView().frame().size.height
        scroll.contentView().scrollToPoint_(NSMakePoint(0, max(0, content_h - clip_h)))
        scroll.reflectScrolledClipView_(scroll.contentView())

        bg.addSubview_(scroll)
        history_window.setContentView_(bg)
        history_window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    NSOperationQueue.mainQueue().addOperationWithBlock_(_show)


def create_menubar():
    global status_item, toggle_helper
    toggle_helper = ToggleHelper.alloc().init()
    status_bar = NSStatusBar.systemStatusBar()
    status_item = status_bar.statusItemWithLength_(NSVariableStatusItemLength)
    status_item.setAutosaveName_("FreeWispr")
    status_item.setVisible_(True)
    status_item.button().setImage_(create_mic_image("idle"))
    rebuild_menu()


def update_menubar_icon(state="idle"):
    from Foundation import NSOperationQueue
    def _update():
        img = create_mic_image(state)
        status_item.button().setImage_(img)
    NSOperationQueue.mainQueue().addOperationWithBlock_(_update)


def notify(message):
    # Escape backslashes and quotes to prevent AppleScript injection
    safe = str(message).replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run([
        "osascript", "-e",
        f'display notification "{safe}" with title "Free Wispr"'
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


active_stream = None


def do_start():
    """Open mic stream and start recording. Always called from a background thread."""
    global recording, audio_frames, active_stream
    if recording or processing:
        return
    try:
        active_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_audio_callback
        )
        active_stream.start()
    except Exception as e:
        _log(f"Stream open error: {e}")
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

    # Close mic stream — releases the mic so macOS indicator disappears
    global active_stream
    try:
        if active_stream:
            active_stream.stop()
            active_stream.close()
            active_stream = None
    except Exception as e:
        _log(f"Stream close error: {e}")

    _log("Stopping...")

    play_sound("Pop")

    try:
        if not frames:
            _log("No frames")
            return

        _log(f"Concatenating {len(frames)} frames...")
        audio = np.concatenate(frames, axis=0)
        del frames  # free memory early
        rms = np.sqrt(np.mean(audio ** 2))
        duration = len(audio) / SAMPLE_RATE
        _log(f"Stopped. RMS={rms:.6f} Duration={duration:.1f}s")
        if rms < 0.0005:
            _log("Too quiet")
            return

        update_menubar_icon("processing")
        _log("Converting audio...")
        audio_int16 = (audio * 32767).astype(np.int16)
        del audio  # free memory early

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_write(f, SAMPLE_RATE, audio_int16)
            tmp_path = f.name
        del audio_int16  # free memory early

        # Save backup before API call
        backup_dir = os.path.expanduser("~/.local/groq-whisper-app/backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.wav")
        try:
            import shutil
            shutil.copy2(tmp_path, backup_path)
            _log(f"Backup saved: {backup_path}")
        except Exception as e:
            _log(f"Backup failed: {e}")

        try:
            _log("Sending to Groq...")
            try:
                text = transcribe_groq(tmp_path)
                _log(f"Groq: '{text}'")
            except Exception as e:
                _log(f"Groq failed: {e}")
                notify("Groq down, using HF fallback")
                text = transcribe_huggingface(tmp_path)
                _log(f"HF: '{text}'")

            if text:
                text = clean_prompt(text)
                _log(f"Cleaned: '{text}'")
                paste_text(text)
                add_to_history(text)
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
        _log("Reset to idle")


def toggle_recording():
    """Toggle between recording and not recording."""
    with state_lock:
        is_rec = recording
        is_proc = processing

    _log(f"Toggle (recording={is_rec} processing={is_proc})")

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
                _log("Reset to idle (forced)")
        threading.Thread(target=_deadline, daemon=True).start()
    elif not is_proc:
        threading.Thread(target=do_start, daemon=True).start()
    # If processing, ignore the tap


poll_fn_was_down = False
poll_fn_down_time = 0
poll_fn_had_other = False
last_toggle_time = 0
poll_count = 0

# Double-tap fn detection
DOUBLE_TAP_WINDOW = 0.35
last_fn_up_time = 0
pending_single_tap_timer = None


def handle_fn_tap():
    """Called when fn is tapped. Distinguishes single vs double tap."""
    global last_fn_up_time, pending_single_tap_timer

    now = time.time()
    since_last_up = now - last_fn_up_time
    last_fn_up_time = now

    if pending_single_tap_timer is not None and pending_single_tap_timer.is_alive():
        # Second tap within window — cancel pending toggle, show history instead
        pending_single_tap_timer.cancel()
        pending_single_tap_timer = None
        threading.Thread(target=show_history_picker, daemon=True).start()
        return

    # First tap — schedule toggle after window expires
    pending_single_tap_timer = threading.Timer(DOUBLE_TAP_WINDOW, toggle_recording)
    pending_single_tap_timer.daemon = True
    pending_single_tap_timer.start()


def poll_fn_key():
    """Called by NSTimer every 0.05s on the main thread."""
    global poll_fn_was_down, poll_fn_down_time, poll_fn_had_other, last_toggle_time, poll_count

    try:
        poll_count += 1
        # Log heartbeat every 60s (1200 ticks at 0.05s)
        if poll_count % 1200 == 0:
            _log(f"poll heartbeat #{poll_count}")

        flags = NSEvent.modifierFlags()
        fn_is_down = bool(flags & NSFunctionKeyMask)

        if fn_is_down and not poll_fn_was_down:
            _log("poll: fn DOWN")
            poll_fn_down_time = time.time()
            poll_fn_had_other = False
        elif not fn_is_down and poll_fn_was_down:
            elapsed = time.time() - poll_fn_down_time
            since_last = time.time() - last_toggle_time
            other_mods = flags & 0xFFFF0000 & ~NSFunctionKeyMask
            _log(f"poll: fn UP elapsed={elapsed:.3f} since_last={since_last:.3f}")
            if not poll_fn_had_other and not other_mods and elapsed < 0.5 and since_last > 0.1:
                last_toggle_time = time.time()
                handle_fn_tap()
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

    load_history()

    if HF_API_KEY:
        _log("HF fallback ready")
    else:
        _log("WARNING: No HF_API_KEY, no fallback")

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    # Disable App Nap — prevents macOS from suspending timers/monitors
    import Foundation
    activity = Foundation.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        0x00FFFFFF,  # NSActivityUserInitiatedAllowingIdleSystemSleep + all flags
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
    # Add timer to common run loop modes so it fires even during menu tracking
    NSRunLoop.currentRunLoop().addTimer_forMode_(_fn_timer, Foundation.NSRunLoopCommonModes)

    # 2. NSEvent monitors as backup (may silently die but works when alive)
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
                    if not evt_state["fn_other"] and elapsed < 0.5 and since_last > 0.1:
                        last_toggle_time = time.time()
                        handle_fn_tap()
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
