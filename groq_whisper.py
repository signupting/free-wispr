"""
Groq Whisper - Tap fn to start/stop recording.
Menubar icon changes state. Uses Groq Whisper API with Hugging Face fallback.
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
from groq import Groq
from scipy.io.wavfile import read as wav_read
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
# Auto-stop a runaway recording before it exceeds provider size limits
# (Groq caps ~25MB ≈ 13min at 16kHz mono; a forgotten recording past that
# 413s and freezes the app). 8 min = ~15.4MB, safely under both providers.
MAX_RECORDING_SECONDS = 480
# Warn this many seconds before the auto-stop fires.
RECORDING_WARN_LEAD_SECONDS = 30
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

groq_client = None

# Adaptive provider selection.
# Which transcription backend works depends on the network: with the VPN ON,
# Groq works but HF may be out of credits; with the VPN OFF, Groq is region-
# blocked (403) but HF works. So instead of a fixed primary, we remember which
# provider succeeded last and try it first. One failure when you toggle the
# VPN, zero waiting after that.
PROVIDER_STATE_PATH = os.path.expanduser(
    "~/.local/groq-whisper-app/last_provider.txt")


def _load_preferred_provider():
    try:
        with open(PROVIDER_STATE_PATH) as f:
            val = f.read().strip()
            if val in ("groq", "hf"):
                return val
    except Exception:
        pass
    return "groq"


def _save_preferred_provider(name):
    try:
        with open(PROVIDER_STATE_PATH, "w") as f:
            f.write(name)
    except Exception:
        pass

# State
recording = False
processing = False
audio_frames = []
stream = None
status_item = None
state_lock = threading.Lock()
# Per-recording warn/auto-stop timers; cancelled when a recording ends so a
# stale timer from an earlier recording can't fire during a later one.
_recording_timers = []


def _cancel_recording_timers():
    global _recording_timers
    for t in _recording_timers:
        t.cancel()
    _recording_timers = []

FN_KEYCODE = 63

# Trigger key: Right Option (works on third-party Bluetooth keyboards, which
# often don't emit the fn / Function modifier that Apple keyboards do).
# 0x40 is the device-dependent "right alt" bit inside modifierFlags(); keycode
# 61 is Right Option in flagsChanged events.
TRIGGER_MASK = 0x40
TRIGGER_KEYCODE = 61
# Other modifiers that, if held with the trigger, mean "don't toggle" — Shift,
# Control, Command, Function (device-independent bits). Option is excluded
# because it's our trigger.
OTHER_MODS_MASK = 0x20000 | 0x40000 | 0x100000 | 0x800000


def create_mic_image(state="idle"):
    from AppKit import NSImageSymbolConfiguration

    # Deliberately NOT a mic glyph. macOS's own mic-in-use privacy indicator
    # sits in the same menu bar using "mic"/"mic.fill", so a mic here reads as
    # a duplicate icon and you can't tell at a glance which one is Free Wispr
    # or whether it's actually running. "waveform" is unmistakably ours; the
    # shape stays constant across states so only colour changes.
    if state == "recording":
        symbol = "waveform"
        color = NSColor.colorWithRed_green_blue_alpha_(0.9, 0.15, 0.15, 1.0)
    elif state == "processing":
        symbol = "waveform"
        color = NSColor.colorWithRed_green_blue_alpha_(1.0, 0.6, 0.0, 1.0)
    else:
        symbol = "waveform"
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
        os.chmod(HISTORY_PATH, 0o600)
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


# ---------------------------------------------------------------- overlay --
# A floating pill above every app showing recording/transcribing state. The
# menu-bar icon already changes colour, but it's one small glyph among a dozen
# others — you can't tell at a glance whether you're actually recording, and a
# dictation you thought was running but wasn't costs a whole re-speak.
#
# Deliberately passive: borderless (so it can never become key), click-through,
# and shown with orderFrontRegardless() so it never steals focus from whatever
# you're dictating into.
# Pill width is computed per state from the text — a fixed width left
# "Transcribing…" and "Recording 0:04" sitting off-centre with dead space on
# the right. Layout is [PAD][dot][GAP][text][PAD], centred on screen.
OVERLAY_H = 54.0
OVERLAY_PAD = 20.0
OVERLAY_GAP = 10.0
OVERLAY_DOT = 13.0
OVERLAY_BOTTOM_MARGIN = 120.0   # clear of the Dock

overlay_window = None
overlay_dot = None
overlay_label = None
overlay_timer = None
_overlay_started_at = 0.0
_overlay_blink = True

# Collection-behaviour bits. Imported by name where available, with literal
# fallbacks — same pragmatism as the raw setMaterial_(7) above.
try:
    from AppKit import (NSWindowCollectionBehaviorCanJoinAllSpaces,
                        NSWindowCollectionBehaviorStationary,
                        NSWindowCollectionBehaviorFullScreenAuxiliary)
except ImportError:  # pragma: no cover
    NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0
    NSWindowCollectionBehaviorStationary = 1 << 4
    NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8
NS_STATUS_WINDOW_LEVEL = 25  # above NSFloatingWindowLevel(3), below screensaver


def _build_overlay():
    """Create the overlay window once. Main thread only."""
    global overlay_window, overlay_dot, overlay_label

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, OVERLAY_BOTTOM_MARGIN, 200, OVERLAY_H),  # sized in _layout_overlay
        NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    win.setLevel_(NS_STATUS_WINDOW_LEVEL)
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setHasShadow_(True)
    win.setIgnoresMouseEvents_(True)      # clicks pass straight through
    win.setReleasedWhenClosed_(False)
    win.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary
    )

    bg = NSVisualEffectView.alloc().initWithFrame_(
        NSMakeRect(0, 0, 200, OVERLAY_H))
    bg.setMaterial_(7)      # HUDWindow — matches the history panel
    bg.setBlendingMode_(0)  # BehindWindow
    bg.setState_(1)
    bg.setWantsLayer_(True)
    bg.layer().setCornerRadius_(OVERLAY_H / 2.0)   # pill
    bg.layer().setMasksToBounds_(True)

    # NSImageView + tinted SF Symbol rather than a CALayer background colour:
    # layer().setBackgroundColor_ needs a CGColor, and PyObjC wraps that as an
    # untyped pointer (ObjCPointerWarning) it doesn't memory-manage. Same
    # palette-configuration trick create_mic_image() already uses.
    from AppKit import NSImageView
    dot = NSImageView.alloc().initWithFrame_(
        NSMakeRect(OVERLAY_PAD, (OVERLAY_H - OVERLAY_DOT) / 2, OVERLAY_DOT, OVERLAY_DOT))
    bg.addSubview_(dot)

    label = NSTextField.alloc().initWithFrame_(
        NSMakeRect(OVERLAY_PAD + OVERLAY_DOT + OVERLAY_GAP, (OVERLAY_H - 22) / 2, 120, 22))
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setFont_(NSFont.systemFontOfSize_(14))
    label.setTextColor_(NSColor.whiteColor())
    bg.addSubview_(label)

    win.setContentView_(bg)
    overlay_window, overlay_dot, overlay_label = win, dot, label


def _layout_overlay(text):
    """Set the label, then size the pill to fit it and re-centre on screen.

    Called on every text change (including each 0.5s tick) so the pill stays
    symmetric as the elapsed clock widens from 0:09 to 0:10 to 10:00.
    """
    overlay_label.setStringValue_(text)
    overlay_label.sizeToFit()
    text_w = overlay_label.frame().size.width
    win_w = OVERLAY_PAD + OVERLAY_DOT + OVERLAY_GAP + text_w + OVERLAY_PAD

    screen = NSScreen.mainScreen().frame()
    x = (screen.size.width - win_w) / 2
    overlay_window.setFrame_display_(
        NSMakeRect(x, OVERLAY_BOTTOM_MARGIN, win_w, OVERLAY_H), True)
    overlay_window.contentView().setFrame_(NSMakeRect(0, 0, win_w, OVERLAY_H))
    # vertical centring: sizeToFit collapses the field to the glyph height
    lf = overlay_label.frame()
    overlay_label.setFrameOrigin_(NSMakePoint(
        OVERLAY_PAD + OVERLAY_DOT + OVERLAY_GAP, (OVERLAY_H - lf.size.height) / 2))


def _dot_image(r, g, b):
    """Filled circle tinted to the same colours as the menu-bar icon states."""
    from AppKit import NSImageSymbolConfiguration
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
        "circle.fill", "status")
    if img is None:
        return None
    color = NSColor.colorWithRed_green_blue_alpha_(r, g, b, 1.0)
    return img.imageWithSymbolConfiguration_(
        NSImageSymbolConfiguration.configurationWithPaletteColors_([color]))


class OverlayHelper(objc.lookUpClass("NSObject")):
    def tick_(self, timer):
        # Blink the dot and tick the elapsed clock while recording.
        global _overlay_blink
        if overlay_window is None:
            return
        _overlay_blink = not _overlay_blink
        elapsed = int(time.time() - _overlay_started_at)
        _layout_overlay(f"Recording   {elapsed // 60}:{elapsed % 60:02d}")
        overlay_dot.setAlphaValue_(1.0 if _overlay_blink else 0.3)


overlay_helper = None


def _overlay_stop_timer():
    global overlay_timer
    if overlay_timer is not None:
        overlay_timer.invalidate()
        overlay_timer = None


def update_overlay(state):
    """Show/hide the floating pill. Main thread only — called from _update()."""
    global overlay_timer, overlay_helper, _overlay_started_at, _overlay_blink

    if state == "idle":
        _overlay_stop_timer()
        if overlay_window is not None:
            overlay_window.orderOut_(None)
        return

    if overlay_window is None:
        _build_overlay()
    if overlay_helper is None:
        overlay_helper = OverlayHelper.alloc().init()

    if state == "recording":
        _overlay_started_at = time.time()
        _overlay_blink = True
        overlay_dot.setImage_(_dot_image(0.9, 0.15, 0.15))
        overlay_dot.setAlphaValue_(1.0)
        _layout_overlay("Recording   0:00")
        _overlay_stop_timer()
        overlay_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.5, overlay_helper, "tick:", None, True)
        NSRunLoop.currentRunLoop().addTimer_forMode_(overlay_timer, "NSRunLoopCommonModes")
    else:  # processing
        _overlay_stop_timer()
        overlay_dot.setImage_(_dot_image(1.0, 0.6, 0.0))
        overlay_dot.setAlphaValue_(1.0)
        _layout_overlay("Transcribing…")

    overlay_window.orderFrontRegardless()   # show without taking focus


def update_menubar_icon(state="idle"):
    from Foundation import NSOperationQueue
    def _update():
        img = create_mic_image(state)
        status_item.button().setImage_(img)
        # Overlay rides the same state funnel as the icon, so every existing
        # transition drives it without touching the call sites.
        try:
            update_overlay(state)
        except Exception as e:
            _log(f"Overlay error: {e}")   # never let chrome break dictation
    NSOperationQueue.mainQueue().addOperationWithBlock_(_update)


def notify(message):
    # Escape backslashes and quotes to prevent AppleScript injection
    safe = str(message).replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run([
        "osascript", "-e",
        f'display notification "{safe}" with title "Free Wispr"'
    ], capture_output=True)


# NVIDIA Parakeet TDT 0.6B v2 via MLX. Chosen over whisper-large-v3-turbo because
# it's 2-3x faster on this M1 (a 60s dictation: ~2s vs ~7s), tops the English ASR
# leaderboard for accuracy, has punctuation/capitalization built in, and — unlike
# Whisper — doesn't pad every clip to 30s, so short dictations are near-instant.
# ~1.5s model load, so even a reload under memory pressure barely stalls.
PARAKEET_MODEL_REPO = "mlx-community/parakeet-tdt-0.6b-v2"
# 8-bit quantized copy of the above, built once by tools/build_quantized.py.
# 734MB vs 1.2GB in memory (byte-identical transcriptions), so on this RAM-tight
# 8GB M1 there's far less to reload when the OS evicts it under swap pressure —
# which is what made dictation "only fast sometimes" (fast warm, ~15s cold).
PARAKEET_QUANT_DIR = os.path.expanduser("~/.local/groq-whisper-app/parakeet-8bit")

_parakeet_model = None
_parakeet_lock = threading.Lock()
# Single dedicated worker for ALL MLX work. MLX streams are per-thread, so
# loading the model on one thread and running inference on another throws
# "There is no Stream(cpu, 1) in current thread." Serializing load + every
# transcription onto one thread keeps them together. (max_workers=1 = one
# persistent thread reused for every task.)
_mlx_executor = ThreadPoolExecutor(max_workers=1)


def _install_numba_stub():
    # parakeet_mlx pulls in librosa, which does `from numba import jit`, and the
    # anaconda base env's numba is broken (coverage.types.Tracer mismatch). Stub
    # numba with a no-op jit so the import succeeds without dragging it in.
    import sys
    import types
    if "numba" in sys.modules and getattr(sys.modules["numba"], "_wispr_stub", False):
        return
    stub = types.ModuleType("numba")
    stub._wispr_stub = True

    def _jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(fn):
            return fn
        return deco

    stub.jit = _jit
    stub.njit = _jit
    sys.modules["numba"] = stub


def _get_parakeet():
    # Lazily load + cache the Parakeet model. First call pays ~1.5s; after that
    # the loaded model is reused for every dictation.
    global _parakeet_model
    if _parakeet_model is not None:
        return _parakeet_model
    with _parakeet_lock:
        if _parakeet_model is not None:
            return _parakeet_model
        _install_numba_stub()
        import json
        import mlx.core as mx
        import mlx.nn as nn
        import parakeet_mlx
        from parakeet_mlx import parakeet as _pk_mod
        from parakeet_mlx.utils import from_config

        # Bypass Parakeet's ffmpeg-based audio loader (dies with "FFmpeg not in
        # PATH" whenever the app is relaunched without /opt/homebrew/bin, exactly
        # like mlx_whisper did). Our wavs are already 16kHz mono int16 — decode
        # in-process with scipy and trim silence here so chunked long audio gets
        # trimmed too.
        def _wav_load_audio(filename, sampling_rate, dtype=mx.bfloat16):
            _r, _s = wav_read(str(filename))
            _a = _s.astype(np.float32) / 32768.0
            _a = _trim_silence(_a, _r)
            return mx.array(_a)

        _pk_mod.load_audio = _wav_load_audio

        # Prefer the local 8-bit quantized copy (smaller = faster reload under
        # memory pressure). Rebuild the quantized layer structure, then load its
        # weights. Fall back to the full bf16 model from HF if it's missing.
        quant_cfg = os.path.join(PARAKEET_QUANT_DIR, "config.json")
        quant_weights = os.path.join(PARAKEET_QUANT_DIR, "model.safetensors")
        if os.path.exists(quant_cfg) and os.path.exists(quant_weights):
            config = json.load(open(quant_cfg))
            model = from_config(config)
            q = config["quantization"]
            nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
            model.load_weights(quant_weights)
            model.eval()
            _parakeet_model = model
        else:
            _parakeet_model = parakeet_mlx.from_pretrained(PARAKEET_MODEL_REPO)
    return _parakeet_model


# Phrases Whisper hallucinates into silence (YouTube/narration boilerplate it
# saw endlessly in training). If a whole transcription collapses to just one of
# these, it was a silent clip — return nothing rather than the garbage.
_HALLUCINATION_PHRASES = {
    "i'm going to show you how to make a video",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to my channel",
    "you",
    "thank you",
    "bye",
}


def _trim_silence(audio, sr=SAMPLE_RATE):
    # Trim leading/trailing near-silence so Whisper never sees a quiet tail to
    # hallucinate into (the "...I'm going to show you how to make a video" at the
    # end of dictations). Frame-based RMS gate at a fraction of the clip's peak.
    if audio.size == 0:
        return audio
    frame = int(0.02 * sr)  # 20ms frames
    if frame < 1:
        return audio
    n = audio.size // frame
    if n < 2:
        return audio
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return audio
    # Voiced if above 8% of the loudest frame, with a small absolute floor so a
    # dead-quiet clip doesn't "find" voice in noise.
    voiced = np.where(rms > max(peak * 0.08, 0.005))[0]
    if voiced.size == 0:
        return audio
    pad = int(0.15 * sr)  # keep 150ms either side so we don't clip word edges
    start = max(0, voiced[0] * frame - pad)
    end = min(audio.size, (voiced[-1] + 1) * frame + pad)
    return audio[start:end]


# Spoken-filler words to strip. Parakeet transcribes disfluencies faithfully
# ("um", "uh") where Whisper silently dropped them; for dictation the user wants
# the Whisper-style clean output. Kept deliberately narrow — no "hmm"/"mm" (can
# be a meaningful reply) and no word that's ever part of a real word.
_FILLER_RE = re.compile(r"\b(?:um|uh|umm|uhh|erm|er)\b[ ,]*", re.IGNORECASE)


def _strip_fillers(text):
    cleaned = _FILLER_RE.sub("", text)
    # Tidy up artifacts left behind: leading punctuation/space, doubled spaces,
    # and a space before punctuation.
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[\s,.!?]+", "", cleaned).strip()
    # Re-capitalize the first letter and any word that now starts a sentence — a
    # stripped "Um" after a period ("...great. Um so...") exposes a lowercase word.
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        cleaned = re.sub(r"([.!?]\s+)([a-z])",
                         lambda m: m.group(1) + m.group(2).upper(), cleaned)
    return cleaned


def _transcribe_local_impl(tmp_path):
    # On-device ASR via NVIDIA Parakeet (Apple MLX). Offline, free, never
    # throttled/timed-out, and 2-3x faster than the whisper-turbo it replaced.
    # The model is loaded once and cached (see _get_parakeet). Runs on the MLX
    # worker thread only — never call this directly, go through transcribe_local.
    import mlx.core as mx
    model = _get_parakeet()
    # chunk_duration splits long audio (>60s) into overlapping windows that
    # Parakeet's own token-merge stitches back together — without it, a multi-
    # minute clip runs a single un-chunked pass that effectively hangs.
    try:
        result = model.transcribe(tmp_path, chunk_duration=60.0, overlap_duration=10.0)
        return _strip_fillers(result.text.strip())
    finally:
        # THE leak fix. MLX caches GPU buffers per tensor SHAPE, and every new
        # dictation length allocates a fresh set that is never released — so the
        # cache climbs with each distinct clip length until it plateaus around
        # 4.4GB (footprint ~5.3GB), which is what tripped FOOTPRINT_RESTART_MB
        # and made the app appear to "quit itself" after a dictation. Measured
        # 2026-08-31: varied lengths 3s..78s reached 5326MB uncleared vs
        # 873-1582MB cleared, and it crosses the 2800MB threshold at a single
        # 34s clip. Cost is nil — +0.02s on a 3s clip, 0.00s at 8s and above.
        # In `finally` so a transcribe error can't strand a full cache.
        mx.clear_cache()


def transcribe_local(tmp_path):
    # Dispatch to the single MLX worker thread and wait for the result, so model
    # load and inference always share a thread (see _mlx_executor).
    return _mlx_executor.submit(_transcribe_local_impl, tmp_path).result()


def transcribe_groq(tmp_path):
    with open(tmp_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            file=(os.path.basename(tmp_path), f.read()),
            # turbo: several times faster than whisper-large-v3, same English
            # accuracy — fixes the multi-minute waits / timeouts.
            model="whisper-large-v3-turbo",
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
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict):
        return result.get("text", "").strip()
    return str(result).strip()


def clean_prompt(raw_text):
    # Cleanup runs on Groq's LLM. Only attempt it when Groq is the currently
    # working provider, so we don't add a wasted blocked call when Groq is down.
    if _load_preferred_provider() != "groq":
        return raw_text
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


LOG_PATH = os.path.expanduser("~/.local/groq-whisper-app/groq-whisper.log")


def _log(msg):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        new_file = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        if new_file:
            os.chmod(LOG_PATH, 0o600)
    except Exception:
        pass


_start_time = time.time()
# Backstop only — the leak this was written for is FIXED (see the mx.clear_cache()
# call in _transcribe_local_impl). It was MLX's GPU buffer cache: MLX caches
# buffers per tensor shape, so every new dictation *length* allocated a fresh set
# that was never freed, climbing to ~5.3GB. An earlier comment here claimed MLX
# inference was ruled out — that was wrong, it had been measured with MLX's own
# active/cache counters, which don't see IOAccelerator allocations. Use
# `footprint <pid>` and read the "IOAccelerator (graphics)" line instead.
#
# With the fix, peak is ~1.3GB against a ~1.4GB fresh baseline, so this should
# now never fire. It stays as a safety net: if it ever does trip, something new
# is leaking and is worth investigating rather than working around.
# launchd (KeepAlive/SuccessfulExit=false) relaunches on a non-zero exit.
FOOTPRINT_RESTART_MB = 2800
MAX_UPTIME_SECONDS = 48 * 3600


def _phys_footprint_mb():
    # Real memory charge incl. compressed/swapped pages — RSS is useless here
    # (the leaked process showed 30MB RSS but 5.3GB phys_footprint). Shells out
    # to the system `footprint` tool; called at most every ~12min off the main
    # thread so its VM-region walk never stalls the UI.
    try:
        out = subprocess.run(["footprint", str(os.getpid())],
                             capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*MB", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())


active_stream = None  # persistent — opened once at startup, never closed


def open_persistent_stream():
    """Open the mic stream once at app startup and keep it open for the app's lifetime.
    Avoids the CoreAudio close-deadlock on macOS Tahoe entirely."""
    global active_stream
    try:
        active_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_audio_callback
        )
        active_stream.start()
        _log("Persistent mic stream opened")
    except Exception as e:
        _log(f"Persistent stream open error: {e}")
        notify(f"Mic init failed: {str(e)[:60]}")


def do_start():
    """Begin capturing frames. Stream stays open whether or not we're recording."""
    global recording, audio_frames
    if recording or processing:
        _log(f"do_start skipped (recording={recording} processing={processing})")
        return
    if active_stream is None:
        _log("do_start: stream not initialized")
        return
    audio_frames = []
    recording = True
    update_menubar_icon("recording")
    play_sound("Tink")
    _log("Recording started")

    # Safety net: warn 30s before, then auto-stop a runaway/forgotten recording
    # so it can't grow past the provider size limit and freeze the app on a 413.
    def _warn():
        if recording:
            _log(f"Recording warning: {RECORDING_WARN_LEAD_SECONDS}s to cap")
            notify(f"Recording stops in {RECORDING_WARN_LEAD_SECONDS}s "
                   f"({MAX_RECORDING_SECONDS//60} min limit)")
            play_sound("Purr")

    def _auto_stop():
        if recording:
            _log(f"Auto-stop: hit {MAX_RECORDING_SECONDS}s cap")
            notify(f"Recording auto-stopped at {MAX_RECORDING_SECONDS//60} min")
            toggle_recording()

    _cancel_recording_timers()  # clear any stragglers before arming fresh ones
    for delay, fn in (
        (max(0, MAX_RECORDING_SECONDS - RECORDING_WARN_LEAD_SECONDS), _warn),
        (MAX_RECORDING_SECONDS, _auto_stop),
    ):
        timer = threading.Timer(delay, fn)
        timer.daemon = True
        timer.start()
        _recording_timers.append(timer)


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

    # Recording is over — kill its warn/auto-stop timers so they can't fire late.
    _cancel_recording_timers()

    _log("Stopping...")

    # Save backup IMMEDIATELY — before stream close or any other op that could hang
    # so we never lose audio to a hung close.
    backup_path = None
    if frames:
        try:
            audio_early = np.concatenate(frames, axis=0)
            audio_early_int16 = (audio_early * 32767).astype(np.int16)
            backup_dir = os.path.expanduser("~/.local/groq-whisper-app/backups")
            os.makedirs(backup_dir, mode=0o700, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.wav")
            wav_write(backup_path, SAMPLE_RATE, audio_early_int16)
            os.chmod(backup_path, 0o600)
            _log(f"Backup saved early: {backup_path}")
            del audio_early, audio_early_int16
        except Exception as e:
            _log(f"Early backup failed: {e}")

    # Stream is persistent — recording=False already stopped frame capture in the callback.
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

        # Backup already saved above (before stream close) — no duplicate

        try:
            # Local MLX Whisper first — on-device, no network, never throttled.
            # Groq / HF stay as cloud fallbacks only if local somehow fails.
            providers = {
                "local": transcribe_local,
                "groq": transcribe_groq,
                "hf": transcribe_huggingface,
            }
            order = ["local", "groq", "hf"]

            text = None
            used = None
            for i, name in enumerate(order):
                try:
                    _log(f"Transcribing via {name.upper()}...")
                    text = providers[name](tmp_path)
                    _log(f"{name.upper()}: '{text}'")
                    used = name
                    if name != "local":
                        notify(f"Local unavailable — used {name.upper()}")
                    break
                except Exception as e:
                    _log(f"{name.upper()} failed: {e}")
                    if i == len(order) - 1:
                        raise  # all failed -> outer handler notifies

            if text:
                # Cleanup runs on Groq's LLM; only do it when Groq itself just
                # worked, so we never hang on a degraded Groq.
                if used == "groq":
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
            else:
                # Empty result (silence / accidental tap) — tell the user rather
                # than looking like a silent failure. Keep the backup just in case.
                _log("Empty transcription (no speech detected)")
                notify("No speech detected")
                play_sound("Basso")
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
            t.join(timeout=90)
            if t.is_alive():
                global processing
                _log("DEADLINE: 90s exceeded, force reset")
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
        fn_is_down = bool(flags & TRIGGER_MASK)

        if fn_is_down and not poll_fn_was_down:
            _log("poll: trigger DOWN")
            poll_fn_down_time = time.time()
            poll_fn_had_other = False
        elif not fn_is_down and poll_fn_was_down:
            elapsed = time.time() - poll_fn_down_time
            since_last = time.time() - last_toggle_time
            other_mods = flags & OTHER_MODS_MASK
            _log(f"poll: trigger UP elapsed={elapsed:.3f} since_last={since_last:.3f}")
            if not poll_fn_had_other and not other_mods and elapsed < 0.5 and since_last > 0.1:
                last_toggle_time = time.time()
                handle_fn_tap()
        elif fn_is_down:
            other_mods = flags & OTHER_MODS_MASK
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

    # Warm the local Parakeet model in the background so the first real dictation
    # doesn't pay the cold-load cost.
    def _warm():
        model = _get_parakeet()
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel
        silent = mx.zeros(SAMPLE_RATE, dtype=mx.float32)  # 1s of silence
        mel = get_logmel(silent, model.preprocessor_config)
        model.generate(mel)

    def _preload_local():
        try:
            # Warm on the MLX worker thread so the model is loaded on the same
            # thread that will later run transcriptions.
            _mlx_executor.submit(_warm).result()
            _log("Local Parakeet model preloaded")
        except Exception as e:
            _log(f"Local preload failed: {e}")
    threading.Thread(target=_preload_local, daemon=True).start()

    # Keep-warm heartbeat: on this 8GB M1 the OS evicts the model's weights from
    # RAM after a few minutes idle (10GB+ in swap), so a dictation after a gap
    # pays a ~15s cold reload — the "only fast sometimes" symptom. Touching the
    # model every 2 min keeps its pages in the working set so it stays fast.
    # Skipped while recording/processing so it never delays a real dictation.
    KEEP_WARM_SECONDS = 120

    def _keep_warm_loop():
        iters = 0
        while True:
            time.sleep(KEEP_WARM_SECONDS)
            iters += 1
            try:
                if recording or processing:
                    continue
                _mlx_executor.submit(_warm).result()

                # Every ~12min (6 x 120s), check the leak watchdog. Skipped while
                # busy above so we never restart mid-dictation.
                if iters % 6 == 0:
                    footprint = _phys_footprint_mb()
                    uptime_h = (time.time() - _start_time) / 3600
                    _log(f"Health: footprint={footprint}MB uptime={uptime_h:.1f}h")
                    over_mem = footprint is not None and footprint > FOOTPRINT_RESTART_MB
                    over_age = (time.time() - _start_time) > MAX_UPTIME_SECONDS
                    if over_mem or over_age:
                        reason = (f"footprint {footprint}MB > {FOOTPRINT_RESTART_MB}"
                                  if over_mem else f"uptime {uptime_h:.1f}h")
                        _log(f"Watchdog restart ({reason}); launchd will relaunch")
                        os._exit(1)  # non-zero -> KeepAlive relaunches fresh
            except Exception as e:
                _log(f"Keep-warm failed: {e}")
    threading.Thread(target=_keep_warm_loop, daemon=True).start()

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
            if keycode == TRIGGER_KEYCODE:
                flags = event.modifierFlags()
                fn_is_down = bool(flags & TRIGGER_MASK)
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

    open_persistent_stream()
    _log("App started (polling + monitors + App Nap disabled)")

    try:
        app.run()
    except KeyboardInterrupt:
        pass
