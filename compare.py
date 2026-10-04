#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = ["librosa", "numpy", "soundfile"]
# ///
"""Render one clip through several voice-removal methods and build a page for A/B listening.

Open <out>/index.html in a browser: the video plays once and you switch which
soundtrack you hear (keys 0-9) without losing your place.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SHUTUPIFY = Path(__file__).resolve().parent / "shutupify.py"

# (key, label, family, description); the key is a model filename for neural methods
METHODS = [
    ("center_cancel", "Center-channel cancellation", "classical",
     "The old karaoke trick: left minus right. Anything panned dead center (usually the voice) cancels, "
     "but so does anything else in the center, and the result is mono."),
    ("repet_sim", "REPET-SIM", "classical",
     "Rafii & Pardo, 2012. Treats spectral frames that resemble many other frames in the track as background "
     "and the rest as voice. Built for repetitive music."),
    ("htdemucs_ft.yaml", "Demucs v4 (htdemucs_ft)", "neural",
     "Meta, 2022. Hybrid waveform + spectrogram transformer; splits vocals / drums / bass / other."),
    ("UVR-MDX-NET-Inst_HQ_4.onnx", "MDX-Net Inst HQ 4", "neural",
     "UVR community, ~2023. Spectrogram U-Net from the 2021 Music Demixing Challenge lineage."),
    ("MDX23C-8KFFT-InstVoc_HQ_2.ckpt", "MDX23C InstVoc HQ 2", "neural",
     "2023 successor to MDX-Net (Sound Demixing Challenge 2023)."),
    ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "BS-RoFormer (viperx 1297)", "neural",
     "Band-split RoPE transformer (ByteDance, 2023). shutupify's current default."),
    ("bs_roformer_vocals_gabox.ckpt", "BS-RoFormer (Gabox vocals)", "neural",
     "Community fine-tune of BS-RoFormer; highest instrumental SDR in the audio-separator catalogue."),
    ("melband_roformer_instvox_duality_v2.ckpt", "Mel-Band RoFormer (InstVoc Duality v2)", "neural",
     "Mel-scale band-split RoFormer (2023), community fine-tune by unwa."),
    ("mel_band_roformer_kim_ft_unwa.ckpt", "Mel-Band RoFormer (Kim FT)", "neural",
     "Vocal-focused Mel-Band RoFormer; background is the mix minus the predicted vocals."),
]


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def center_cancel(y):
    side = (y[:, 0] - y[:, 1]) / 2
    mid = (y[:, 0] + y[:, 1]) / 2
    return np.stack([side, side], 1), np.stack([mid, mid], 1)


def repet_sim(y, sr):
    # Following librosa's vocal separation example: a soft mask from a nearest-neighbor
    # median filter, computed on the mono magnitude and applied to each channel.
    import librosa

    stfts = [librosa.stft(ch) for ch in y.T]
    S = np.abs(sum(stfts) / len(stfts))
    S_filter = librosa.decompose.nn_filter(
        S, aggregate=np.median, metric="cosine", width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S, S_filter)
    mask_bg = librosa.util.softmask(S_filter, 2 * (S - S_filter), power=2)
    mask_fg = librosa.util.softmask(S - S_filter, 10 * S_filter, power=2)
    bg = [librosa.istft(mask_bg * D, length=len(y)) for D in stfts]
    fg = [librosa.istft(mask_fg * D, length=len(y)) for D in stfts]
    return np.stack(bg, 1), np.stack(fg, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="video file")
    parser.add_argument("-s", "--start", default="0", help="excerpt start (seconds or [hh:]mm:ss)")
    parser.add_argument("-d", "--duration", default="60", help="excerpt length in seconds (default: 60)")
    parser.add_argument("-o", "--out", type=Path, help="output directory (default: comparisons/<input name>)")
    args = parser.parse_args()
    out = args.out or Path(__file__).resolve().parent / "comparisons" / args.input.stem
    out.mkdir(parents=True, exist_ok=True)

    # Re-encode the excerpt so it cuts exactly and plays in any browser; every method starts from it.
    clip = out / "clip.mp4"
    if not clip.exists():
        print(f"Cutting {args.duration}s from {args.input} at {args.start}...", file=sys.stderr)
        ffmpeg("-ss", args.start, "-t", args.duration, "-i", str(args.input), "-map", "0:v:0", "-map", "0:a:0",
               "-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-c:a", "aac", "-b:a", "320k", "-ac", "2", "-ar", "44100", "-movflags", "+faststart", str(clip))
    original = out / "original.wav"
    if not original.exists():
        ffmpeg("-i", str(clip), "-c:a", "pcm_s16le", str(original))

    variants = [{"key": "original", "label": "Original", "family": "reference",
                 "desc": "Unprocessed soundtrack.", "result": original.name, "removed": None}]
    for key, label, family, desc in METHODS:
        result, removed = out / f"{key}.wav", out / f"{key}.voice.wav"
        if result.exists() and removed.exists():
            print(f"[cached] {label}", file=sys.stderr)
        elif family == "classical":
            print(f"[running] {label}", file=sys.stderr)
            y, sr = sf.read(original, dtype="float32")
            bg, fg = center_cancel(y) if key == "center_cancel" else repet_sim(y, sr)
            sf.write(result, bg, sr, subtype="PCM_16")
            sf.write(removed, fg, sr, subtype="PCM_16")
        else:
            print(f"[running] {label}", file=sys.stderr)
            proc = subprocess.run([str(SHUTUPIFY), str(clip), "-m", key, "-o", str(result), "--save-vocals", str(removed)])
            if proc.returncode != 0:
                print(f"[failed] {label}; leaving it out", file=sys.stderr)
                continue
        variants.append({"key": key, "label": label, "family": family, "desc": desc,
                         "result": result.name, "removed": removed.name})

    page = (Path(__file__).resolve().parent / "compare_template.html").read_text()
    page = page.replace("/*VARIANTS*/[]", json.dumps(variants, indent=1))
    page = page.replace("<!--TITLE-->", f"{args.input.name} @ {args.start}s, {args.duration}s")
    (out / "index.html").write_text(page)
    print(f"\nOpen {out / 'index.html'} in a browser.", file=sys.stderr)


if __name__ == "__main__":
    main()
