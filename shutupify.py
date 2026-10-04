#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "audio-separator[gpu]>=0.47; sys_platform == 'linux'",
#     "audio-separator[cpu]>=0.47; sys_platform == 'darwin'",  # torch still uses the Apple GPU (MPS)
#     "audioread",  # imported by audio-separator, but librosa>=1.0 no longer pulls it in
#     "onnxruntime-gpu<1.27; sys_platform == 'linux'",  # 1.27+ is built for CUDA 13; see the torch note below
#     "torch",
#     "torchvision",
# ]
#
# # On Linux, PyPI's default torch wheels target CUDA 13, which needs a newer driver than
# # many machines have. The cu126 build runs on any driver from the CUDA 12 era onward.
# [tool.uv.sources]
# torch = { index = "pytorch-cu126", marker = "sys_platform == 'linux'" }
# torchvision = { index = "pytorch-cu126", marker = "sys_platform == 'linux'" }
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Strip human voices from a video (or audio) file, keeping music and sound effects.

The soundtrack is split into "vocals" and "everything else" by a neural source
separation model (python-audio-separator, which wraps the UVR model zoo), and the
video is remuxed with only the non-vocal part. Video streams are copied untouched.

Some models worth trying with --model:
  model_bs_roformer_ep_317_sdr_12.9755.ckpt   BS-RoFormer (default)
  melband_roformer_instvox_duality_v2.ckpt    Mel-Band RoFormer
  MDX23C-8KFFT-InstVoc_HQ_2.ckpt              MDX23C
  htdemucs_ft.yaml                            Demucs v4 (fine-tuned)
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MODEL_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "shutupify" / "models"
SAMPLE_RATE = 44100  # what the separation models are trained on
AUDIO_ONLY_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def count_audio_streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return len(out.split())


def extract_audio(src, dst, track):
    """Extract one audio track as peak-normalized float WAV; returns the gain that undoes it."""
    # Float so nothing clips or gets requantized. The aresample filter pads any leading
    # gap with silence, so the WAV lines up with the video at t=0.
    ffmpeg("-i", str(src), "-map", f"0:a:{track}", "-af", "aresample=async=1:first_pts=0",
           "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(dst))
    # audio-separator expects input peaking at exactly 0 dBFS: it scales down anything
    # louder (lossy decodes often are), and its MDX-Net path multiplies the output by the
    # input peak whatever it is. Matching that expectation keeps every model at the
    # source's level once the gain is undone.
    audio, _ = sf.read(dst, dtype="float32")
    peak = float(np.abs(audio).max())
    if peak == 0:
        sys.exit(f"error: audio track {track} of {src} is silent")
    sf.write(dst, audio / peak, SAMPLE_RATE, subtype="FLOAT")
    return peak


def separate(wav, workdir, model):
    """Run the separation model and return (voice, background) WAV paths."""
    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=str(workdir),
        model_file_dir=str(MODEL_DIR),
        output_format="WAV",
        sample_rate=SAMPLE_RATE,
        use_soundfile=True,  # keeps the float input's subtype instead of writing 16-bit
        normalization_threshold=1.0,
        log_level=logging.ERROR,
    )
    separator.load_model(model_filename=model)
    stems = {}
    for name in separator.separate(str(wav)):
        path = Path(workdir) / Path(name).name
        label = re.match(rf"{re.escape(wav.stem)}_\((.+?)\)_", path.name).group(1)
        stems[label] = path

    # Two-stem models give vocals + instrumental; multi-stem ones (e.g. Demucs) give
    # vocals + drums/bass/other, so the background is the sum of every non-vocal stem.
    voice = [p for label, p in stems.items() if "vocal" in label.lower()]
    background = [p for label, p in stems.items() if "vocal" not in label.lower()]
    if len(voice) != 1 or not background:
        sys.exit(f"error: can't tell which stems are vocals for model {model}: {sorted(stems)}")
    if len(background) == 1:
        return voice[0], background[0]
    mix = sum(sf.read(p, dtype="float32")[0] for p in background)
    out = Path(workdir) / "background.wav"
    sf.write(out, np.asarray(mix), SAMPLE_RATE, subtype="FLOAT")
    return voice[0], out


def mux(src, audio, gain, dst, track):
    args = ["-i", str(src), "-i", str(audio)]
    if dst.suffix.lower() not in AUDIO_ONLY_EXTS:
        args += ["-map", "0:v?", "-c:v", "copy"]
    args += ["-map", "1:a", "-af", f"volume={gain}", "-b:a", "192k", "-map_metadata", "0", "-map_metadata:s:a:0", f"0:s:a:{track}"]
    if src.suffix.lower() == dst.suffix.lower():
        args += ["-map", "0:s?", "-c:s", "copy"]  # subtitle codecs rarely survive a container change
    ffmpeg(*args, str(dst))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="video or audio file")
    parser.add_argument("-o", "--output", type=Path, help="output file (default: <input>.novoice.<ext>)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"separation model (default: {DEFAULT_MODEL})")
    parser.add_argument("-t", "--audio-track", type=int, default=0, help="which audio track to process (default: 0)")
    parser.add_argument("--save-vocals", type=Path, metavar="FILE", help="also save the removed voice track here")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("error: ffmpeg not found on PATH")
    if not args.input.is_file():
        sys.exit(f"error: no such file: {args.input}")
    n_tracks = count_audio_streams(args.input)
    if args.audio_track >= n_tracks:
        sys.exit(f"error: {args.input} has {n_tracks} audio track(s); can't use track {args.audio_track}")
    output = args.output or args.input.with_suffix(f".novoice{args.input.suffix}")

    with tempfile.TemporaryDirectory(prefix="shutupify-") as tmp:
        tmp = Path(tmp)
        wav = tmp / "audio.wav"
        print(f"Extracting audio from {args.input}...", file=sys.stderr)
        gain = extract_audio(args.input, wav, args.audio_track)
        print(f"Separating voices with {args.model}...", file=sys.stderr)
        voice, background = separate(wav, tmp, args.model)
        mux(args.input, background, gain, output, args.audio_track)
        if args.save_vocals:
            ffmpeg("-i", str(voice), "-af", f"volume={gain}", str(args.save_vocals))
    print(f"Wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
