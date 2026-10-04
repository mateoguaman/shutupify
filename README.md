# shutupify

Strip human voices from a video, keeping the music and sound effects.

```sh
./shutupify.py movie.mp4                  # writes movie.novoice.mp4
```

The soundtrack is split into "vocals" and "everything else" by a neural source
separation model, and the video is remuxed with only the non-vocal part. The
video stream is copied bit for bit.

## Setup

You need [uv](https://docs.astral.sh/uv/) and ffmpeg. uv installs everything else
(PyTorch, the separation library) into a cached environment on first run.

**macOS** (Apple Silicon):

```sh
brew install uv ffmpeg
```

Homebrew also installs Xcode's command line tools, which the first run needs to
compile one dependency. Models run on the Apple GPU via Metal (MPS).

**Ubuntu:**

```sh
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

With an NVIDIA GPU (driver 525 or newer), models run on CUDA; otherwise on the
CPU, which is much slower. On an RTX 4090 the default model runs about 5x faster
than real time.

Then:

```sh
git clone git@github.com:mateoguaman/shutupify.git
cd shutupify
./shutupify.py some_video.mp4
```

The first run takes a few minutes: it downloads PyTorch and the model (~640 MB,
cached in `~/.cache/shutupify/models`).

## Usage

```sh
./shutupify.py movie.mkv -o quiet.mkv            # choose the output name
./shutupify.py movie.mp4 -o soundtrack.wav       # audio only
./shutupify.py movie.mp4 --save-vocals voice.wav # also keep what was removed
./shutupify.py movie.mkv -t 1                    # process the second audio track
./shutupify.py movie.mp4 -m htdemucs_ft.yaml     # use a different model
```

Only the chosen audio track ends up in the output. Surround audio is downmixed to
stereo, since the models are stereo.

## Comparing models

Which model sounds best depends on the material, so `compare.py` renders one
excerpt through several of them, plus two classical methods for reference, and
builds a page to A/B them:

```sh
./compare.py movie.mp4 --start 300 --duration 60
open comparisons/movie/index.html                # xdg-open on Linux
```

The video plays once while you switch soundtracks with the number keys; `v`
toggles between the result and what was removed. Models download on first use
(0.2–0.9 GB each).

## How it works, and why these models

Voice removal is a *source separation* problem. Classical approaches lean on
assumptions about the mix:

- **Center-channel cancellation** subtracts the right channel from the left, which
  cancels anything panned dead center. Voices usually are, but so are bass and
  kick drums, and the result is mono.
- **REPET** and its variants treat whatever repeats as background and the rest as
  voice. That works for loop-based music, not for film.
- **NMF** and other spectrogram factorizations need hand tuning and leave
  audible artifacts.

Neural networks trained on multitrack recordings do far better. The lineage runs
from Spleeter (Deezer, 2019) through Demucs (Meta) and MDX-Net to the band-split
transformers that lead the benchmarks now:
[BS-RoFormer](https://arxiv.org/abs/2309.02612) and
[Mel-Band RoFormer](https://arxiv.org/abs/2310.01809). shutupify uses them
through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator),
which packages the community-trained [UVR](https://github.com/Anjok07/ultimatevocalremovergui)
models. These are trained on music, but they handle speech well.

There is also research on separating film audio into dialogue, music, and effects
([BandIt](https://github.com/kwatcharasupat/bandit), the DnR datasets), but no
ready-to-run models for it yet.
