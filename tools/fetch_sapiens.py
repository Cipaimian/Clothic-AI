"""Download a Meta Sapiens body-part segmentation checkpoint (Model 1).

Sapiens seg checkpoints are TorchScript files hosted on Hugging Face. This
fetches one into ``models/sapiens/`` and prints the path to wire into the
FullBackend (``sapiens_checkpoint=...``).

    pip install huggingface_hub
    python tools/fetch_sapiens.py --size 0.3b
    python tools/fetch_sapiens.py --size 1b --out models/sapiens

Sizes: 0.3b (lightest, good for CPU/start) · 0.6b · 1b · 2b.
NOTE: review Meta's Sapiens license before any non-research/commercial use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Official Meta TorchScript repos, one per size. The exact .pt2 filename embeds
# mIoU/epoch and is auto-detected, so we never hard-code a fragile name.
# Approx sizes: 0.3b 1.36 GB · 0.6b 2.69 GB · 1b 4.72 GB · 2b 8.71 GB.
SIZES = ["0.3b", "0.6b", "1b", "2b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.3b", choices=SIZES)
    ap.add_argument("--repo", default=None, help="override the HF repo id")
    ap.add_argument("--filename", default=None, help="override the exact filename")
    ap.add_argument("--out", default="models/sapiens", type=Path)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install the downloader first:  pip install huggingface_hub")
        return 1

    repo = args.repo or f"facebook/sapiens-seg-{args.size}-torchscript"
    filename = args.filename
    if filename is None:
        weights = [f for f in list_repo_files(repo) if f.endswith((".pt2", ".pt"))]
        if not weights:
            print(f"No .pt2 checkpoint found in {repo}")
            return 1
        filename = weights[0]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo}/{filename} ...")
    try:
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(args.out))
    except Exception as exc:  # noqa: BLE001 - surface a clear hint
        print(f"Download failed: {exc}\n"
              f"Check the exact repo/filename on Hugging Face and pass --repo/--filename. "
              f"You may also need: huggingface-cli login")
        return 1

    print(f"\nSaved: {path}")
    print("Wire it in, e.g.:")
    print("  ClothicPipeline(backend='full', backend_kwargs={")
    print(f"      'sapiens_checkpoint': '{path}',")
    print("      'person_weights': 'legacy/yolov8n.pt',")
    print("      'garment_weights': 'legacy/runs/detect/train/weights/best.pt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
