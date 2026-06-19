# Clothic AI - Datasets & the Unified Garment Set

The garment detector (Model 2) trains on **one merged YOLO dataset** built by
`tools/dataset_unifier.py` from any number of sources, each remapped into the
canonical vocabulary in `ontology/garment_classes.yaml`. This lets us grow the
data (and fix the under-represented violation classes) without ever retraining
on mismatched labels.

```
sources (Roboflow / DeepFashion2 / Fashionpedia)
   └─ *_to_canon mapping ─► canonical 13 classes ─► data/merged_garment/ (YOLO)
                                                     ▲ balanced, rare-class-first
```

Build / rebuild:
```bash
python tools/dataset_unifier.py --config configs/dataset_sources.yaml --dry-run  # counts only
python tools/dataset_unifier.py --config configs/dataset_sources.yaml            # writes data/merged_garment
```
Enable a source by setting `enabled: true` in `configs/dataset_sources.yaml`
once its files are on disk.

---

## Why we need more data (current gap)

The Student Dress Code export alone is small and **lopsided exactly where it
matters**: the violation classes are the rarest.

| class | instances | role |
|-------|-----------|------|
| shoe / longsleeved / shirt | 138–263 | compliant - fine |
| **sleeveless** | 107 | violation - thin |
| **shorts / skirt / ripped_jeans / mini-skirt** | 46–65 | violation - **too few** |
| **dress** | 46 | violation - too few |

Rule of thumb: ≥150–200 instances/class for a usable detector. The sources below
fill the gap. (Length/rip/crop are intentionally **not** detector classes - Sapiens
pixel-exposure and FashionSigLIP decide those - so the detector only needs robust
TYPE coverage: sleeveless / shirt / shorts / pants / skirt / dress / shoe.)

---

## Source 1 - DeepFashion2  *(biggest accuracy lever)*

491K images, 13 categories. Directly fills our gaps: `vest`+`sling` → **sleeveless**,
plus thousands of **shorts / skirt / dress** instances. Mapping:
`ontology/mappings/deepfashion2_to_canon.yaml`.

1. Request access (Google Form) and download from the official repo:
   <https://github.com/switchablenorms/DeepFashion2>. The zips are password
   protected - the password is given in the form's confirmation.
2. Unzip so the layout is:
   ```
   deepfashion2/
     train/{image,annos}/        # annos/*.json hold category_id + bounding_box
     validation/{image,annos}/
   ```
   (Place `deepfashion2/` at the project root, or edit `path:` in the config.)
3. In `configs/dataset_sources.yaml` set `deepfashion2: enabled: true`.
4. Rebuild. The `cap_per_class: 1200` balancer keeps the merge from being swamped
   by DeepFashion2's common classes. **Tip:** on this CPU-only machine you do not
   need all 491K images - the cap naturally subsamples; expect a few-thousand-image
   merged set that trains overnight.

## Source 2 - Fashionpedia  *(attributes + extra TYPE coverage)*

48K images, COCO format, **no login** (HuggingFace / CVDF mirror). Also the
intended ground-truth for FashionSigLIP attribute training later. Mapping:
`ontology/mappings/fashionpedia_to_canon.yaml`.

1. Download images + `instances_attributes_{train,val}2020.json` from
   <https://github.com/cvdfoundation/fashionpedia> (or HF
   `detection-datasets/fashionpedia`).
2. Layout:
   ```
   fashionpedia/
     images/                          # all jpgs
     instances_attributes_train2020.json
     instances_attributes_val2020.json
   ```
3. Set `fashionpedia: enabled: true` and rebuild.

## Source 3+ - more Roboflow dress-code exports  *(quick top-ups)*

Any YOLO export works with `kind: yolo`. Drop a `<name>_to_canon.yaml` with a
`by_name:` block (source label → canonical name) and add a source entry. Good
candidates: "Clothing Detection" (16 cls, has SleevelessShirt) and other
dress-code sets on Roboflow Universe.

---

## After adding data

```bash
python tools/dataset_unifier.py --config configs/dataset_sources.yaml   # rebuild
python training/stage_c_garment/train.py --data training/stage_c_garment/data.yaml \
    --model yolo11s.pt --epochs 100 --imgsz 640 --device cpu           # real run
```
Then point `configs/pipeline.yaml: full.garment_weights` at the new
`runs/.../weights/best.pt` and re-run `scripts/verify_full_backend.py`.

> Note: `yolo11s.pt` (stronger base than `yolo11n`) auto-downloads from GitHub,
> which has timed out on this network before. If so, fetch it once manually or
> fall back to the local `yolo11n.pt`.
