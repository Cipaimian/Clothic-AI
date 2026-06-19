"""Merge heterogeneous fashion datasets into ONE canonical YOLO dataset.

Every source (Roboflow YOLO exports, DeepFashion2, Fashionpedia) is remapped into
the canonical garment vocabulary (``ontology/garment_classes.yaml``) and written
out as a single YOLO detection dataset the detector can train on directly. A
greedy, rare-class-first balancer caps the dominant classes so the scarce
violation classes (sleeveless / shorts / skirt / dress) are not drowned out.

Provenance is preserved in filenames (``<source>__<stem>``) so any annotation can
be traced back and re-mapped later.

Usage::

    python tools/dataset_unifier.py --config configs/dataset_sources.yaml
    python tools/dataset_unifier.py --config configs/dataset_sources.yaml --dry-run

Add a dataset by dropping a ``*_to_canon.yaml`` mapping and a source entry in the
config; enable it once downloaded. See docs/DATASETS.md.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

# An annotation = (canonical_class_id, (cx, cy, w, h) normalized YOLO bbox).
Ann = tuple[int, tuple[float, float, float, float]]
# A record = one image plus its annotations, tagged with split + source.
Record = dict  # {source, split, image_path: Path, anns: list[Ann]}


# --------------------------------------------------------------------------- #
# Canonical vocabulary
# --------------------------------------------------------------------------- #
def load_canonical(path: Path) -> tuple[dict[str, int], list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["classes"]
    id_to_name = {int(k): v for k, v in data.items()}
    names = [id_to_name[i] for i in range(len(id_to_name))]
    name_to_id = {n: i for i, n in id_to_name.items()}
    return name_to_id, names


# --------------------------------------------------------------------------- #
# Per-source loaders -> yield Record
# --------------------------------------------------------------------------- #
def _yolo_names(root: Path) -> dict[int, str]:
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    return {int(k): v for k, v in names.items()} if isinstance(names, dict) \
        else {i: n for i, n in enumerate(names)}


def load_yolo(src: dict, canon: dict[str, int]) -> list[Record]:
    """Roboflow/YOLO export. ``identity`` sources already use canonical names."""
    root = Path(src["path"])
    id_to_name = _yolo_names(root)
    if src.get("identity"):
        remap = {n: n for n in id_to_name.values()}
    else:
        remap = yaml.safe_load(Path(src["mapping"]).read_text(encoding="utf-8"))["by_name"]

    out: list[Record] = []
    for split_dir, split in (("train", "train"), ("valid", "valid"), ("test", "test")):
        labels = root / split_dir / "labels"
        images = root / split_dir / "images"
        if not labels.exists():
            continue
        for lf in labels.glob("*.txt"):
            anns: list[Ann] = []
            for line in lf.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                name = id_to_name.get(int(p[0]))
                canon_name = remap.get(name)
                cid = canon.get(canon_name) if canon_name else None
                if cid is None:
                    continue
                anns.append((cid, (float(p[1]), float(p[2]), float(p[3]), float(p[4]))))
            img = images / (lf.stem + ".jpg")
            if not img.exists():  # some exports use .png
                alt = list(images.glob(lf.stem + ".*"))
                img = alt[0] if alt else img
            if img.exists():
                out.append({"source": src["name"], "split": split,
                            "image_path": img, "anns": anns})
    return out


def _to_yolo_bbox(x1, y1, w_box, h_box, iw, ih) -> tuple[float, float, float, float]:
    """Pixel (x,y,w,h top-left) -> normalized (cx,cy,w,h), clamped to [0,1]."""
    cx = (x1 + w_box / 2) / iw
    cy = (y1 + h_box / 2) / ih
    return (min(max(cx, 0), 1), min(max(cy, 0), 1),
            min(max(w_box / iw, 0), 1), min(max(h_box / ih, 0), 1))


def load_deepfashion2(src: dict, canon: dict[str, int]) -> list[Record]:
    """DeepFashion2: per-image annos/<stem>.json with item*.category_id + bbox."""
    from PIL import Image  # lazy

    root = Path(src["path"])
    by_id = {int(k): v for k, v in
             yaml.safe_load(Path(src["mapping"]).read_text(encoding="utf-8"))["by_id"].items()}
    out: list[Record] = []
    for split_dir, split in (("train", "train"), ("validation", "valid")):
        anno_dir = root / split_dir / "annos"
        img_dir = root / split_dir / "image"
        if not anno_dir.exists():
            continue
        for jf in anno_dir.glob("*.json"):
            data = json.loads(jf.read_text(encoding="utf-8"))
            img = img_dir / (jf.stem + ".jpg")
            if not img.exists():
                continue
            items = [v for k, v in data.items() if k.startswith("item")]
            if not items:
                continue
            iw, ih = Image.open(img).size
            anns: list[Ann] = []
            for it in items:
                cid = canon.get(by_id.get(int(it["category_id"])))
                if cid is None:
                    continue
                x1, y1, x2, y2 = it["bounding_box"]
                anns.append((cid, _to_yolo_bbox(x1, y1, x2 - x1, y2 - y1, iw, ih)))
            if anns:
                out.append({"source": src["name"], "split": split,
                            "image_path": img, "anns": anns})
    return out


def load_fashionpedia(src: dict, canon: dict[str, int]) -> list[Record]:
    """Fashionpedia: COCO instances json (images[] + annotations[] + categories[])."""
    root = Path(src["path"])
    by_name = yaml.safe_load(Path(src["mapping"]).read_text(encoding="utf-8"))["by_name"]
    out: list[Record] = []
    for json_glob, split in (("*train*.json", "train"), ("*val*.json", "valid")):
        for cj in root.glob(json_glob):
            coco = json.loads(cj.read_text(encoding="utf-8"))
            cat = {c["id"]: c["name"] for c in coco["categories"]}
            imgs = {im["id"]: im for im in coco["images"]}
            per_img: dict[int, list[Ann]] = {}
            for a in coco["annotations"]:
                cid = canon.get(by_name.get(cat.get(a["category_id"])))
                if cid is None:
                    continue
                im = imgs[a["image_id"]]
                x, y, w_box, h_box = a["bbox"]
                per_img.setdefault(a["image_id"], []).append(
                    (cid, _to_yolo_bbox(x, y, w_box, h_box, im["width"], im["height"])))
            for img_id, anns in per_img.items():
                img = root / "images" / Path(imgs[img_id]["file_name"]).name
                if img.exists():
                    out.append({"source": src["name"], "split": split,
                                "image_path": img, "anns": anns})
    return out


LOADERS = {"yolo": load_yolo, "deepfashion2": load_deepfashion2,
           "fashionpedia": load_fashionpedia}


# --------------------------------------------------------------------------- #
# Balancing
# --------------------------------------------------------------------------- #
def balance_train(records: list[Record], cap: int | None, seed: int) -> list[Record]:
    """Greedy rare-class-first selection so dominant classes don't drown rare ones.

    Process images that contain the globally-rarest classes first; keep an image
    only while at least one of its classes is still under ``cap``. All annotations
    of a kept image are retained (never partially-labelled).
    """
    if cap is None:
        return records
    freq: Counter[int] = Counter(c for r in records for c, _ in r["anns"])
    rng = random.Random(seed)
    rng.shuffle(records)
    # rarest-first: order by the least-frequent class an image carries
    records.sort(key=lambda r: min((freq[c] for c, _ in r["anns"]), default=10**9))

    kept: list[Record] = []
    count: Counter[int] = Counter()
    for r in records:
        classes = {c for c, _ in r["anns"]}
        if any(count[c] < cap for c in classes):
            kept.append(r)
            for c in classes:
                count[c] += sum(1 for cc, _ in r["anns"] if cc == c)
    return kept


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_dataset(records: list[Record], names: list[str], out: Path,
                  dry_run: bool) -> dict:
    per_split_classes: dict[str, Counter] = {s: Counter() for s in ("train", "valid", "test")}
    per_source = Counter()
    provenance: dict[str, str] = {}
    if not dry_run:
        for split in ("train", "valid", "test"):
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(records):
        split = r["split"]
        # Short, index-based stem -> avoids Windows MAX_PATH (260) blowups from
        # long Roboflow hash names. Original path kept in provenance.json.
        stem = f"{r['source']}__{i:06d}"
        per_source[r["source"]] += 1
        for c, _ in r["anns"]:
            per_split_classes[split][c] += 1
        if dry_run:
            continue
        dst_img = out / split / "images" / (stem + r["image_path"].suffix.lower())
        shutil.copy(r["image_path"], dst_img)
        lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in r["anns"]]
        (out / split / "labels" / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        provenance[stem] = str(r["image_path"])

    if not dry_run:
        data_yaml = {
            "path": str(out.resolve()).replace("\\", "/"),
            "train": "train/images", "val": "valid/images", "test": "test/images",
            "nc": len(names), "names": {i: n for i, n in enumerate(names)},
        }
        (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        (out / "provenance.json").write_text(json.dumps(provenance, indent=0), encoding="utf-8")

    return {
        "images": {s: sum(1 for r in records if r["split"] == s) for s in ("train", "valid", "test")},
        "instances_per_class": {
            s: {names[c]: n for c, n in sorted(cc.items())} for s, cc in per_split_classes.items()
        },
        "images_per_source": dict(per_source),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report counts, copy nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    canon, names = load_canonical(Path(cfg["canonical"]))
    out = Path(cfg["output"])
    bal = cfg.get("balance", {})

    all_records: list[Record] = []
    for src in cfg["sources"]:
        if not src.get("enabled", True):
            print(f"[skip] {src['name']} (disabled)")
            continue
        if not Path(src["path"]).exists():
            print(f"[skip] {src['name']} (path not found: {src['path']})")
            continue
        loaded = LOADERS[src["kind"]](src, canon)
        # External datasets are extra TRAINING data; evaluation stays on the
        # campus dress-code val/test. `to_split` forces a source into one split.
        if src.get("to_split"):
            for r in loaded:
                r["split"] = src["to_split"]
        print(f"[load] {src['name']}: {len(loaded)} images")
        all_records += loaded

    if not all_records:
        print("No records loaded -- enable at least one source.")
        return 1

    train = [r for r in all_records if r["split"] == "train"]
    other = [r for r in all_records if r["split"] != "train"]
    train = balance_train(train, bal.get("cap_per_class"), bal.get("seed", 0))
    final = train + other

    summary = write_dataset(final, names, out, args.dry_run)
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        print(f"\nMerged dataset written to {out}/  (data.yaml ready for training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
