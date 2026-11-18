# Tolk

Live Norwegian → English subtitles for everything playing on your PC. Captures
speaker output directly through WASAPI loopback, so it works with Teams, Zoom,
YouTube and anything else without a virtual audio cable. Runs entirely offline
once the model is downloaded.

## Run it from source

Python 3.10–3.12 (3.13 support in some of the audio wheels is still patchy).

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tolk.py
```

First launch downloads the Whisper model into a `models` folder beside the
script — around 500 MB for `small`. After that it never touches the network.

Press **Start**, then play anything. Nothing else to configure: it follows
whatever your current default playback device is.

## Build the exe

```
build.bat
```

Produces `dist\Tolk\Tolk.exe`. Make a desktop shortcut to that. Keep the whole
`Tolk` folder together — the one-folder build is far more reliable than
`--onefile` here, because CTranslate2 ships native libraries that don't
survive being unpacked to a temp directory on every launch.

The build is large (roughly 1 GB with the model). If you want it smaller, delete
the `models` folder before building and let it download on first run instead.

## Tuning

Everything worth changing sits at the top of `tolk.py`.

**`MODEL_SIZE`** is the main dial. `small` keeps up with speech on a normal CPU.
`medium` is clearly better on Norwegian and worth it if you have a GPU or don't
mind a few seconds of lag. `large-v3` is best but needs a real GPU. If you have
an NVIDIA card the app finds it automatically and switches to fp16.

**`VOCABULARY`** matters more than you'd expect. Put your colleagues' names,
your employer, project names and recurring jargon in there — it's passed to
Whisper as context and fixes most proper-noun mangling.

**`SHOW_SOURCE`** displays the Norwegian under each English line. It costs a
second pass over the same audio, so turn it off if you're running tight on CPU.
It's useful at first as a sanity check: if the Norwegian is wrong, the English
above it can't be right.

**`SILENCE_TAIL`** controls latency. Text appears after a pause, so 0.7 s means
roughly a sentence of delay. Lowering it to 0.4 makes it snappier but chops
mid-sentence more often, which hurts translation quality. Raising it to 1.0
gives cleaner sentences and more lag.

## Why Whisper rather than the built-in Windows models

Whisper's `translate` task goes from Norwegian audio to English text in a single
pass, so nothing is lost to a separate translation step, and it's substantially
more robust on regional dialect than the consumer speech models — it was trained
on a lot of messy real-world audio. It still isn't magic on broad nordnorsk, but
`medium` handles it far better than `small` does. If output looks poor, try
bumping the model size before assuming the audio routing is broken.

## Troubleshooting

**"No loopback device found"** — WASAPI loopback is Windows-only. On a Mac
you'd need BlackHole as an intermediate device instead.

**Nothing appears but the level meter moves** — audio is arriving but the model
is still loading, or your segments are below `MIN_SEGMENT`. Check the status
line at the bottom.

**Level meter is dead** — Windows changed default playback device after the app
started. Stop and Start again.

**Very slow** — you're on CPU with a big model. Drop to `small`, or set
`SHOW_SOURCE = False` to halve the work per segment.
