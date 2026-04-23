import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import predict_ensemble as pe

# Resolve defaults relative to this script so commands work from any cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PATIENT_DIR = _SCRIPT_DIR / "patient_data"
_DEFAULT_LABELS_CSV = _DEFAULT_PATIENT_DIR / "labels_ac.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AD (A) vs HC (C) EDF subjects with LEADv2 ensemble."
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=str(_DEFAULT_LABELS_CSV),
        help="CSV path with at least columns: participant_id, Group (default: patient_data/labels_ac.csv next to this script).",
    )
    parser.add_argument(
        "--edf_dir",
        type=str,
        default=str(_DEFAULT_PATIENT_DIR),
        help="Directory containing EDF files named like sub-001_task-eyesclosed_eeg.edf (default: patient_data/).",
    )
    parser.add_argument(
        "--edf_suffix",
        type=str,
        default="_task-eyesclosed_eeg.edf",
        help="Suffix appended to participant_id to build EDF filename.",
    )
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
        help="Comma-separated seed folder names used for ensemble.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--sampling_rate", type=int, default=200, choices=[200, 100, 50])
    parser.add_argument("--seq_len", type=int, default=400)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--low_cut", type=float, default=0.5)
    parser.add_argument("--high_cut", type=float, default=45.0)
    parser.add_argument("--notch", type=float, default=50.0, help="Set 0 to disable notch.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_avg_ref", action="store_true")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional per-subject prediction CSV output path.",
    )
    return parser.parse_args()


def load_labels(labels_csv: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"participant_id", "Group"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")
        for row in reader:
            pid = row["participant_id"].strip()
            group = row["Group"].strip().upper()
            if group in {"A", "C"}:
                labels[pid] = group
    if not labels:
        raise ValueError("No A/C rows found in labels CSV.")
    return labels


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "A" and p == "A")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "A" and p == "C")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "C" and p == "A")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "C" and p == "C")
    n = len(y_true)

    acc = (tp + tn) / n if n else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0.0

    return {
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
        "n": float(n),
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    labels_csv = Path(args.labels_csv)
    edf_dir = Path(args.edf_dir)
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    checkpoint_root = Path(args.checkpoint_root)
    seed_folders = [s.strip() for s in args.seed_folders.split(",") if s.strip()]
    notch = None if args.notch == 0 else args.notch

    labels = load_labels(labels_csv)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[info] Device: {device}")
    print(f"[info] Evaluating {len(labels)} A/C subjects from: {labels_csv}")

    models = pe.load_models(
        checkpoint_root=checkpoint_root,
        seed_folders=seed_folders,
        device=device,
        seq_len=args.seq_len,
    )

    y_true: List[str] = []
    y_pred: List[str] = []
    missing_files: List[str] = []
    per_subject_rows: List[Tuple[str, str, str, float, float, int]] = []

    for pid, true_group in sorted(labels.items()):
        edf_path = edf_dir / f"{pid}{args.edf_suffix}"
        if not edf_path.exists():
            missing_files.append(str(edf_path))
            continue

        data_tc, fs = pe.preprocess_edf(
            edf_path=edf_path,
            target_fs=args.sampling_rate,
            low_cut=args.low_cut,
            high_cut=args.high_cut,
            notch=notch,
            use_avg_ref=(not args.no_avg_ref),
        )
        segments = pe.segment_signal(data_tc=data_tc, seq_len=args.seq_len, step=args.step)
        out = pe.infer_ensemble(
            models=models,
            segments_ntc=segments,
            sampling_rate=fs,
            batch_size=args.batch_size,
            device=device,
        )
        pred_label = "A" if int(out["subject_label"][0]) == 1 else "C"
        ad_prob = float(out["subject_prob"][1])
        hc_prob = float(out["subject_prob"][0])

        y_true.append(true_group)
        y_pred.append(pred_label)
        per_subject_rows.append((pid, true_group, pred_label, ad_prob, hc_prob, int(segments.shape[0])))

        print(
            f"[{pid}] true={true_group} pred={pred_label} "
            f"AD_prob={ad_prob:.4f} HC_prob={hc_prob:.4f} segments={segments.shape[0]}"
        )

    metrics = compute_metrics(y_true, y_pred)

    print("\n========== A vs C Evaluation ==========")
    print("Confusion Matrix (rows=true [A,C], cols=pred [A,C])")
    print(f"[[TP={int(metrics['tp'])}, FN={int(metrics['fn'])}], [FP={int(metrics['fp'])}, TN={int(metrics['tn'])}]]")
    print(f"N={int(metrics['n'])}")
    print(
        "Accuracy={:.4f}  Sensitivity(AD Recall)={:.4f}  Specificity(HC Recall)={:.4f}  "
        "Precision={:.4f}  F1={:.4f}".format(
            metrics["accuracy"],
            metrics["sensitivity"],
            metrics["specificity"],
            metrics["precision"],
            metrics["f1"],
        )
    )
    print("=======================================\n")

    if missing_files:
        print("[warn] Missing EDF files:")
        for path in missing_files:
            print(f"  - {path}")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["participant_id", "true_group", "pred_group", "ad_prob", "hc_prob", "n_segments"])
            writer.writerows(per_subject_rows)
        print(f"[ok] Per-subject predictions saved to: {out_csv}")


if __name__ == "__main__":
    main()
