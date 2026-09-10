#!/usr/bin/env python3
import argparse
from pathlib import Path

def parse_int_stem(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return None

def collect_frame_indices(frames_dir: Path):
    pngs = sorted(frames_dir.glob("*.png"))
    idxs = []
    for p in pngs:
        idx = parse_int_stem(p)
        if idx is not None:
            idxs.append(idx)
    # unique + sorted
    return sorted(set(idxs))

def write_list(path: Path, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in indices:
            f.write(f"{i}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", type=str, required=True, help="Root folder containing subject folders")
    ap.add_argument("--subjects", type=str, nargs="+", required=True,
                    help="Subject folder names under root_path")
    ap.add_argument("--frames_dir_name", type=str, default="frames",
                    help="Name of the frames directory inside each subject (default: frames)")
    args = ap.parse_args()

    root = Path(args.root_path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"root_path does not exist: {root}")

    for subj in args.subjects:
        subj_dir = root / subj
        frames_dir = subj_dir / args.frames_dir_name

        if not subj_dir.exists():
            print(f"[SKIP] Subject folder not found: {subj_dir}")
            continue
        if not frames_dir.exists():
            print(f"[SKIP] Frames folder not found: {frames_dir}")
            continue

        indices = collect_frame_indices(frames_dir)
        if not indices:
            print(f"[WARN] No numeric *.png frames found in: {frames_dir}")
            continue

        write_list(subj_dir / "frame_list_all.txt", indices)
        write_list(subj_dir / "frame_list_train.txt", indices)
        write_list(subj_dir / "frame_list_test.txt", indices)

        print(f"[OK] {subj}: wrote {len(indices)} indices (min={indices[0]}, max={indices[-1]})")

if __name__ == "__main__":
    main()
