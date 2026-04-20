import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import torch

from data_provider.uea import normalize_batch_ts
from models import LEADv2


# Training-time channel order used for LEADv2 fine-tuning.
TARGET_CHANNELS: List[str] = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


# Common alias mapping to normalize EDF channel names.
CHANNEL_ALIASES: Dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "A1": "M1",
    "A2": "M2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEADv2 AD-vs-HC ensemble inference from EDF.")
    parser.add_argument("--edf_path", type=str, required=True, help="Path to a single EDF file.")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="./checkpoints/LEADv2/finetune/LEADv2/P-Base-F-ADFTD-AD-vs-HC",
        help="Root folder containing seed checkpoint subfolders.",
    )
    parser.add_argument(
        "--seed_folders",
        type=str,
        default="nh8_el12_dm128_df256_seed41,nh8_el12_dm128_df256_seed43,nh8_el12_dm128_df256_seed44",
        help="Comma-separated seed folder names to ensemble.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size over segments.")
    parser.add_argument("--sampling_rate", type=int, default=200, choices=[200, 100, 50], help="Target sampling rate.")
    parser.add_argument("--seq_len", type=int, default=400, help="Window length in samples.")
    parser.add_argument("--step", type=int, default=200, help="Sliding window step in samples.")
    parser.add_argument("--low_cut", type=float, default=0.5, help="Bandpass low cutoff (Hz).")
    parser.add_argument("--high_cut", type=float, default=45.0, help="Bandpass high cutoff (Hz).")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch frequency (Hz), use 0 to disable.")
    parser.add_argument(
        "--line_freq",
        type=float,
        default=50.0,
        help="Alias for notch frequency; kept for convenience.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--no_avg_ref", action="store_true", help="Disable average re-referencing.")
    return parser.parse_args()


def _canonical_name(ch_name: str) -> str:
    name = ch_name.strip()
    # Drop prefixes often seen in EDF exports.
    for prefix in ("EEG ", "EEG-", "EEG_"):
        if name.upper().startswith(prefix.strip().upper()):
            name = name[len(prefix):]
            break
    # Remove common suffixes.
    for suffix in ("-REF", "-LE", "-RE", "_REF", "_LE", "_RE"):
        if name.upper().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = name.replace(" ", "")
    name = CHANNEL_ALIASES.get(name, name)
    return name


def _build_model_config(seq_len: int, num_class: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="finetune",
        output_attention=False,
        patch_len=50,
        stride=50,
        enc_in=19,
        seq_len=seq_len,
        d_model=128,
        n_heads=8,
        e_layers=12,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        channel_names=",".join(TARGET_CHANNELS),
        montage_name="standard_1005",
        augmentations="none",
        num_class=num_class,
    )


def _load_single_model(model_path: Path, device: torch.device, seq_len: int) -> torch.nn.Module:
    cfg = _build_model_config(seq_len=seq_len, num_class=2)
    model = LEADv2.Model(cfg).to(device)
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older torch versions that do not support weights_only.
        ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    # Handle DataParallel / SWA style checkpoints.
    cleaned = {}
    for k, v in ckpt.items():
        if k == "n_averaged":
            continue
        new_key = k
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"[warn] Unexpected keys for {model_path.name}: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing:
        print(f"[warn] Missing keys for {model_path.name}: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    model.eval()
    return model


def load_models(checkpoint_root: Path, seed_folders: Sequence[str], device: torch.device, seq_len: int) -> List[torch.nn.Module]:
    models: List[torch.nn.Module] = []
    for folder in seed_folders:
        model_path = checkpoint_root / folder / "checkpoint.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        models.append(_load_single_model(model_path, device=device, seq_len=seq_len))
        print(f"[ok] Loaded checkpoint: {model_path}")
    return models


def preprocess_edf(
    edf_path: Path,
    target_fs: int,
    low_cut: float,
    high_cut: float,
    notch: float,
    use_avg_ref: bool,
) -> Tuple[np.ndarray, int]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    rename_map = {ch: _canonical_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    if notch and notch > 0:
        raw.notch_filter(freqs=[notch], verbose="ERROR")
    raw.filter(l_freq=low_cut, h_freq=high_cut, verbose="ERROR")
    if use_avg_ref:
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    present = set(raw.ch_names)
    missing = [ch for ch in TARGET_CHANNELS if ch not in present]
    if missing:
        raise ValueError(
            "EDF is missing required channels for LEADv2: "
            + ", ".join(missing)
            + ".\nProvide EDF with 10-20 channels matching training setup."
        )

    raw.pick(TARGET_CHANNELS)
    # Enforce exact channel order expected by the model.
    raw.reorder_channels(TARGET_CHANNELS)
    raw.resample(sfreq=target_fs, verbose="ERROR")
    data = raw.get_data()  # (C, T)
    data = data.T.astype(np.float32)  # (T, C)
    return data, int(target_fs)


def segment_signal(data_tc: np.ndarray, seq_len: int, step: int) -> np.ndarray:
    total_t = data_tc.shape[0]
    if total_t < seq_len:
        raise ValueError(
            f"Signal too short after preprocessing: T={total_t}, required at least seq_len={seq_len}"
        )
    starts = list(range(0, total_t - seq_len + 1, step))
    segments = np.stack([data_tc[s : s + seq_len] for s in starts], axis=0)  # (N, T, C)
    segments = normalize_batch_ts(segments)
    return segments.astype(np.float32)


@torch.no_grad()
def infer_ensemble(
    models: Sequence[torch.nn.Module],
    segments_ntc: np.ndarray,
    sampling_rate: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    n_segments = segments_ntc.shape[0]
    all_probs: List[np.ndarray] = []

    fs_batch_template = None

    for i in range(0, n_segments, batch_size):
        batch_np = segments_ntc[i : i + batch_size]
        x = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)  # (B, T, C)
        b = x.shape[0]
        padding_mask = torch.ones((b, x.shape[1]), device=device, dtype=torch.float32)
        if fs_batch_template is None or fs_batch_template.shape[0] != b:
            fs_batch_template = torch.full((b,), float(sampling_rate), device=device, dtype=torch.float32)
        fs = fs_batch_template

        model_probs = []
        for model in models:
            logits = model(x, padding_mask, None, None, fs, None)
            probs = torch.softmax(logits, dim=1)
            model_probs.append(probs)
        ensemble_probs = torch.stack(model_probs, dim=0).mean(dim=0)  # (B, 2)
        all_probs.append(ensemble_probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # (N, 2)
    pred_labels = probs.argmax(axis=1)  # segment-level
    # Subject-level: average probabilities across all segments.
    subject_prob = probs.mean(axis=0)  # (2,)
    subject_label = int(subject_prob.argmax())
    return {
        "segment_probs": probs,
        "segment_pred_labels": pred_labels,
        "subject_prob": subject_prob,
        "subject_label": np.array([subject_label], dtype=np.int64),
    }


def print_report(edf_path: Path, inference: Dict[str, np.ndarray]) -> None:
    label_map = {0: "HC", 1: "AD"}
    seg_pred = inference["segment_pred_labels"]
    probs = inference["segment_probs"]
    subject_prob = inference["subject_prob"]
    subject_label = int(inference["subject_label"][0])

    n = len(seg_pred)
    ad_ratio = float((seg_pred == 1).sum()) / n
    hc_ratio = float((seg_pred == 0).sum()) / n

    print("\n========== Ensemble Inference Report ==========")
    print(f"EDF: {edf_path}")
    print(f"Segments: {n}")
    print(f"Segment vote ratio -> HC: {hc_ratio:.3f}, AD: {ad_ratio:.3f}")
    print(
        f"Subject probability (avg over segments) -> "
        f"HC: {subject_prob[0]:.4f}, AD: {subject_prob[1]:.4f}"
    )
    print(f"Final prediction: {label_map[subject_label]} ({subject_prob[subject_label]:.4f})")
    print("==============================================\n")


def main() -> None:
    args = parse_args()
    edf_path = Path(args.edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    checkpoint_root = Path(args.checkpoint_root)
    seed_folders = [s.strip() for s in args.seed_folders.split(",") if s.strip()]
    if not seed_folders:
        raise ValueError("No seed folders provided.")

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[info] Using device: {device}")

    notch = args.notch if args.notch is not None else args.line_freq
    if notch == 0:
        notch = None

    data_tc, fs = preprocess_edf(
        edf_path=edf_path,
        target_fs=args.sampling_rate,
        low_cut=args.low_cut,
        high_cut=args.high_cut,
        notch=notch,
        use_avg_ref=(not args.no_avg_ref),
    )
    segments = segment_signal(data_tc=data_tc, seq_len=args.seq_len, step=args.step)
    print(f"[info] Preprocessed signal shape: {data_tc.shape}, segmented into {segments.shape[0]} windows.")

    models = load_models(
        checkpoint_root=checkpoint_root,
        seed_folders=seed_folders,
        device=device,
        seq_len=args.seq_len,
    )
    inference = infer_ensemble(
        models=models,
        segments_ntc=segments,
        sampling_rate=fs,
        batch_size=args.batch_size,
        device=device,
    )
    print_report(edf_path=edf_path, inference=inference)


if __name__ == "__main__":
    main()
