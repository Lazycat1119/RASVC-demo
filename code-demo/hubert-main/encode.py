import argparse
import numpy as np
from pathlib import Path

from scipy.io import wavfile
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample
from dataset.f0_pred import RMVPEF0Predictor


def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("./", f"hubert_soft", source="local").cuda().eval()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if True:  # not os.path.exists(out_path.with_suffix(".npy")):
            wav, sr = torchaudio.load(in_path)

            wav = resample(wav, sr, 16000)
            wav1 = wav.unsqueeze(0).numpy()
            wav = wav.unsqueeze(0).cuda()

            with torch.inference_mode():
                units = hubert.units(wav)
            wavfile.write(
                out_path,
                16000,
                (wav1 * np.iinfo(np.int16).max).astype(np.int16)
            )

            np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())
            f0_predictor = RMVPEF0Predictor(hop_length=320, sampling_rate=16000, dtype=torch.float32,device=None,threshold=0.05)
            f0,uv = f0_predictor.compute_f0_uv(
                wav
            )
            np.save(out_path.with_suffix(".f0.npy"), np.asanyarray((f0,uv),dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models (HuBERT-Soft or HuBERT-Discrete)",
        choices=["soft", "discrete"],
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
