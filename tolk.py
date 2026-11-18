"""
Tolk - live Norwegian -> English subtitles for anything playing on your PC.

Captures the audio coming out of your speakers (WASAPI loopback, so it works
with Teams, Zoom, YouTube, anything at all), splits it on natural pauses, and
runs each segment through Whisper's translate task. Fully offline once the
model has been downloaded.

    python tolk.py
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None

try:
    import soxr
except ImportError:
    soxr = None


# --------------------------------------------------------------------------
# Settings you may want to change
# --------------------------------------------------------------------------

# tiny / base / small / medium / large-v3
# "small" is roughly real-time on a modern CPU. "medium" is noticeably better
# on Norwegian and still fine if you have a GPU or a fast recent laptop.
MODEL_SIZE = "small"

SOURCE_LANGUAGE = "no"      # "no" bokmal, "nn" nynorsk, "sv", "da", or None to autodetect
SHOW_SOURCE = False         # set to False to eliminate second pass and double speed

# Names, jargon and place names that Whisper keeps mangling. This is fed to the
# model as context and genuinely helps with proper nouns.
VOCABULARY = "Skjervøy, Troms, Tromsø, kommune, Teams, prosjekt, budsjett"

SILENCE_TAIL = 0.7          # seconds of quiet that ends a segment (captures full sentences)
MIN_SEGMENT = 0.8           # ignore anything shorter than this
MAX_SEGMENT = 14.0          # force a cut here so long monologues still appear
PREROLL = 0.35              # audio kept from before speech was detected

TARGET_RATE = 16000         # what Whisper wants


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

GROUND = "#0E1A24"
PANEL = "#14232F"
RULE = "#23394B"
TEXT = "#EAF2F5"
SOURCE = "#7C96A8"
DIM = "#5A7386"
LIVE = "#E9B872"
BAD = "#E2795F"


def app_dir():
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(exe_dir, "models")):
            return exe_dir
        parent_dir = os.path.dirname(os.path.dirname(exe_dir))
        if os.path.exists(os.path.join(parent_dir, "models")):
            return parent_dir
        return exe_dir
    return os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------

class LoopbackCapture(threading.Thread):
    """Reads the default output device via WASAPI loopback and emits speech
    segments as float32 mono at 16 kHz."""

    def __init__(self, out_queue, status):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.status = status
        self.running = threading.Event()
        self.level = 0.0
        self.device_name = "-"
        self._pa = None

    def find_loopback(self, pa):
        if pyaudio is None:
            raise RuntimeError(
                "pyaudiowpatch is not installed. Run: pip install pyaudiowpatch"
            )
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return speakers
        for candidate in pa.get_loopback_device_info_generator():
            if speakers["name"] in candidate["name"]:
                return candidate
        raise RuntimeError(
            "No loopback device found. WASAPI loopback needs Windows and the "
            "pyaudiowpatch package."
        )

    def stop(self):
        self.running.clear()

    def run(self):
        if pyaudio is None:
            self.status("pyaudiowpatch is not installed", error=True)
            return

        self.running.set()
        try:
            self._pa = pyaudio.PyAudio()
            device = self.find_loopback(self._pa)
            rate = int(device["defaultSampleRate"])
            channels = int(device["maxInputChannels"])
            self.device_name = device["name"]
            chunk = 2048
            chunk_seconds = chunk / rate

            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=device["index"],
            )
        except Exception as exc:
            self.status(f"Audio device error: {exc}", error=True)
            if self._pa:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
            return

        self.status(f"Listening to {self.device_name}")

        noise_floor = 0.004
        preroll_len = max(1, int(PREROLL / chunk_seconds))
        preroll = []
        collected = []
        in_speech = False
        quiet_for = 0.0
        speech_for = 0.0

        try:
            while self.running.is_set():
                try:
                    raw = stream.read(chunk, exception_on_overflow=False)
                except Exception:
                    continue

                if not raw:
                    continue

                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                if channels > 1:
                    remainder = len(samples) % channels
                    if remainder != 0:
                        samples = samples[:-remainder]
                    if len(samples) == 0:
                        continue
                    samples = samples.reshape(-1, channels).mean(axis=1)

                if len(samples) == 0:
                    continue

                mean_sq = np.mean(samples ** 2)
                if np.isnan(mean_sq):
                    continue

                rms = float(np.sqrt(mean_sq) + 1e-9)
                self.level = rms
                threshold = max(0.0035, noise_floor * 3.2)

                if rms > threshold:
                    if not in_speech:
                        in_speech = True
                        collected = list(preroll)
                        speech_for = 0.0
                    collected.append(samples)
                    speech_for += chunk_seconds
                    quiet_for = 0.0
                else:
                    # Track the noise floor only while nobody is talking.
                    noise_floor = 0.98 * noise_floor + 0.02 * rms
                    if in_speech:
                        collected.append(samples)
                        quiet_for += chunk_seconds

                preroll.append(samples)
                if len(preroll) > preroll_len:
                    preroll.pop(0)

                ended = in_speech and (quiet_for >= SILENCE_TAIL or speech_for >= MAX_SEGMENT)
                if ended:
                    in_speech = False
                    if speech_for >= MIN_SEGMENT and collected:
                        audio = np.concatenate(collected)
                        self.out_queue.put(self.to_16k(audio, rate))
                    collected = []
                    quiet_for = 0.0
                    speech_for = 0.0
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    @staticmethod
    def to_16k(audio, rate):
        if rate == TARGET_RATE:
            return audio
        if soxr is not None:
            return soxr.resample(audio, rate, TARGET_RATE)
        n = int(len(audio) * TARGET_RATE / rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, n, dtype=np.float64),
            np.arange(len(audio), dtype=np.float64),
            audio,
        ).astype(np.float32)


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

class Transcriber(threading.Thread):
    def __init__(self, in_queue, emit, status):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.emit = emit
        self.status = status
        self.running = threading.Event()
        self.model = None

    def load(self):
        # pyrefly: ignore [missing-import]
        from faster_whisper import WhisperModel

        device, compute = "cpu", "int8"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device, compute = "cuda", "float16"
        except Exception:
            pass

        self.status(f"Loading {MODEL_SIZE} on {device}, first run downloads it...")
        self.model = WhisperModel(
            MODEL_SIZE,
            device=device,
            compute_type=compute,
            download_root=os.path.join(app_dir(), "models"),
        )
        self.status(f"Ready - {MODEL_SIZE} on {device}")

    def run_task(self, audio, task):
        if self.model is None:
            return ""
        segments, _ = self.model.transcribe(
            audio,
            task=task,
            language=SOURCE_LANGUAGE,
            beam_size=1,
            temperature=0.0,
            initial_prompt=VOCABULARY or None,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def stop(self):
        self.running.clear()

    def run(self):
        self.running.set()
        try:
            self.load()
        except Exception as exc:
            self.status(f"Could not load the model: {exc}", error=True)
            return

        while self.running.is_set():
            try:
                audio = self.in_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            # Drop backlog rather than fall further behind.
            while self.in_queue.qsize() > 3:
                try:
                    self.in_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                english = self.run_task(audio, "translate")
                if not english:
                    continue
                norwegian = self.run_task(audio, "transcribe") if SHOW_SOURCE else ""
                self.emit(english, norwegian)
            except Exception as exc:
                self.status(f"Transcription failed: {exc}", error=True)


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------

class Tolk(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tolk - Norwegian to English")
        self.geometry("880x520")
        self.minsize(520, 260)
        self.configure(bg=GROUND)

        self.audio_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.capture = None
        self.transcriber = None
        self.started_at = None
        self.lines = []
        self.font_size = 22

        self.build()
        self.after(80, self.pump)

    # -- layout ------------------------------------------------------------

    def build(self):
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x")
        tk.Frame(self, bg=RULE, height=1).pack(fill="x")

        tk.Label(
            bar, text="TOLK  /  NO\u2192EN", bg=PANEL, fg=TEXT,
            font=("Consolas", 10), padx=14, pady=11,
        ).pack(side="left")

        self.dot = tk.Canvas(bar, width=9, height=9, bg=PANEL, highlightthickness=0)
        self.dot.create_oval(1, 1, 8, 8, fill=DIM, outline="")
        self.dot.pack(side="left", padx=(0, 7))

        self.state_label = tk.Label(
            bar, text="Stopped", bg=PANEL, fg=DIM, font=("Consolas", 9),
        )
        self.state_label.pack(side="left")

        self.toggle = self.button(bar, "Start", self.toggle_running, accent=True)
        self.toggle.pack(side="right", padx=(6, 14), pady=8)
        self.button(bar, "Save", self.save).pack(side="right", padx=3, pady=8)
        self.button(bar, "Clear", self.clear).pack(side="right", padx=3, pady=8)
        self.button(bar, "A-", lambda: self.resize(-2)).pack(side="right", padx=3, pady=8)
        self.button(bar, "A+", lambda: self.resize(2)).pack(side="right", padx=3, pady=8)

        self.pin = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bar, text="On top", variable=self.pin, command=self.set_pin,
            bg=PANEL, fg=DIM, selectcolor=PANEL, activebackground=PANEL,
            activeforeground=TEXT, font=("Consolas", 9), bd=0, highlightthickness=0,
        ).pack(side="right", padx=8, pady=8)

        wrap = tk.Frame(self, bg=GROUND)
        wrap.pack(fill="both", expand=True)

        self.text = tk.Text(
            wrap, bg=GROUND, fg=TEXT, bd=0, highlightthickness=0,
            wrap="word", padx=24, pady=20, spacing1=2, spacing3=14,
            cursor="arrow", state="disabled",
        )
        scroll = tk.Scrollbar(wrap, command=self.text.yview, bd=0,
                              troughcolor=GROUND, bg=RULE, width=10)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.apply_tags()

        tk.Frame(self, bg=RULE, height=1).pack(fill="x")
        foot = tk.Frame(self, bg=PANEL)
        foot.pack(fill="x")
        self.meter = tk.Canvas(foot, width=90, height=4, bg=PANEL, highlightthickness=0)
        self.meter.pack(side="left", padx=14, pady=12)
        self.status_label = tk.Label(
            foot, text="Press Start. Everything playing on this PC will be captured.",
            bg=PANEL, fg=DIM, font=("Segoe UI", 9), anchor="w",
        )
        self.status_label.pack(side="left", fill="x", expand=True)

    def button(self, parent, label, command, accent=False):
        return tk.Button(
            parent, text=label, command=command,
            bg=PANEL, fg=LIVE if accent else DIM,
            activebackground=RULE, activeforeground=TEXT,
            font=("Consolas", 9), bd=1, relief="solid",
            highlightbackground=RULE, padx=12 if accent else 8, pady=3,
            cursor="hand2",
        )

    def apply_tags(self):
        self.text.tag_configure(
            "en", font=("Georgia", self.font_size), foreground=TEXT,
            lmargin1=0, lmargin2=0, spacing3=4,
        )
        self.text.tag_configure(
            "no", font=("Segoe UI", max(9, self.font_size - 11)), foreground=SOURCE,
            spacing3=16,
        )
        self.text.tag_configure(
            "stamp", font=("Consolas", 9), foreground=DIM, spacing3=2,
        )

    # -- controls ----------------------------------------------------------

    def set_pin(self):
        self.attributes("-topmost", self.pin.get())

    def resize(self, delta):
        self.font_size = max(13, min(44, self.font_size + delta))
        self.apply_tags()

    def clear(self):
        self.lines = []
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def save(self):
        if not self.lines:
            return
        name = "transcript-" + datetime.now().strftime("%Y-%m-%d-%H%M") + ".txt"
        path = os.path.join(app_dir(), name)
        with open(path, "w", encoding="utf-8") as handle:
            for stamp, english, norwegian in self.lines:
                handle.write(f"{stamp}\n{english}\n")
                if norwegian:
                    handle.write(f"{norwegian}\n")
                handle.write("\n")
        self.status(f"Saved to {path}")

    def toggle_running(self):
        if self.capture and self.capture.running.is_set():
            self.stop()
        else:
            self.start()

    def start(self):
        self.started_at = self.started_at or time.time()
        self.transcriber = Transcriber(self.audio_queue, self.emit, self.status)
        self.transcriber.start()
        self.capture = LoopbackCapture(self.audio_queue, self.status)
        self.capture.start()
        self.toggle.configure(text="Stop", fg=TEXT)
        self.state_label.configure(text="Listening", fg=LIVE)
        self.dot.itemconfigure(1, fill=LIVE)

    def stop(self):
        if self.capture:
            self.capture.stop()
        if self.transcriber:
            self.transcriber.stop()
        self.toggle.configure(text="Start", fg=LIVE)
        self.state_label.configure(text="Stopped", fg=DIM)
        self.dot.itemconfigure(1, fill=DIM)
        self.status("Stopped")

    # -- thread-safe plumbing ---------------------------------------------

    def emit(self, english, norwegian):
        self.ui_queue.put(("line", english, norwegian))

    def status(self, message, error=False):
        self.ui_queue.put(("status", message, error))

    def pump(self):
        while True:
            try:
                kind, a, b = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self.append(a, b)
            else:
                self.status_label.configure(text=a, fg=BAD if b else DIM)

        if self.capture and self.capture.running.is_set():
            self.draw_meter(self.capture.level)

        self.after(80, self.pump)

    def append(self, english, norwegian):
        elapsed = time.time() - (self.started_at or time.time())
        stamp = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        self.lines.append((stamp, english, norwegian))

        at_bottom = self.text.yview()[1] > 0.92
        self.text.configure(state="normal")
        self.text.insert("end", stamp + "\n", "stamp")
        self.text.insert("end", english + "\n", "en")
        if norwegian:
            self.text.insert("end", norwegian + "\n", "no")
        self.text.configure(state="disabled")
        if at_bottom:
            self.text.see("end")

    def draw_meter(self, level):
        self.meter.delete("all")
        width = int(min(1.0, level * 14) * 90)
        self.meter.create_rectangle(0, 1, 90, 4, fill=RULE, outline="")
        if width:
            self.meter.create_rectangle(0, 1, width, 4, fill=LIVE, outline="")


if __name__ == "__main__":
    if pyaudio is None:
        print("pyaudiowpatch is missing. Run: pip install -r requirements.txt")
    app = Tolk()
    app.mainloop()
