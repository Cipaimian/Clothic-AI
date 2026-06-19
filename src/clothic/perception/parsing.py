"""Model 1 - human body parsing → pixel-measured skin exposure per region.

Default model is **Meta Sapiens** body-part segmentation (Goliath 28-class
vocabulary). The vocabulary is ideal for this project because it labels *bare
anatomy* and *clothing* as separate classes: a covered arm is ``Upper Clothing``
pixels, a bare arm is ``Right/Left Upper Arm`` pixels. So per body region:

    exposure(region) = skin_pixels / (skin_pixels + clothing_pixels)

i.e. how much of the visible body region is uncovered. A garment "covers" a
region exactly when its exposure is low. This is the auditable signal the rule
engine consumes -- no model judges modesty, geometry + policy do.

Both parsers are lazy-imported; the base package needs neither torch nor
transformers. ``SegformerParser`` is kept as a lighter fallback. The exposure
math (``region_exposure_ratio``) is a pure numpy function and is unit-tested
without any model.
"""

from __future__ import annotations

import numpy as np

# --- Sapiens Goliath 28-class vocabulary -------------------------------------
# Order matches Meta's released body-part segmentation head.
GOLIATH_CLASSES: list[str] = [
    "Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand",
    "Left Lower Arm", "Left Lower Leg", "Left Shoe", "Left Sock", "Left Upper Arm",
    "Left Upper Leg", "Lower Clothing", "Right Foot", "Right Hand", "Right Lower Arm",
    "Right Lower Leg", "Right Shoe", "Right Sock", "Right Upper Arm", "Right Upper Leg",
    "Torso", "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth",
    "Tongue",
]

# Bare-skin anatomy that, when visible, signals exposure.
SKIN_CLASSES = {
    "Face Neck", "Torso",
    "Left Upper Arm", "Right Upper Arm", "Left Lower Arm", "Right Lower Arm",
    "Left Hand", "Right Hand",
    "Left Upper Leg", "Right Upper Leg", "Left Lower Leg", "Right Lower Leg",
    "Left Foot", "Right Foot",
}
# Anything worn that covers the body.
CLOTHING_CLASSES = {
    "Apparel", "Upper Clothing", "Lower Clothing",
    "Left Shoe", "Right Shoe", "Left Sock", "Right Sock",
}


def class_ids(label_map: dict[int, str], names: set[str]) -> set[int]:
    """Resolve a set of class names to ids against a model's label map."""
    wanted = {n.lower() for n in names}
    return {i for i, name in label_map.items() if name.lower() in wanted}


def region_exposure_ratio(
    seg: np.ndarray,
    skin_ids: set[int],
    clothing_ids: set[int],
    regions: dict[str, tuple[int, int, int, int]],
) -> dict[str, float]:
    """exposure = skin / (skin + clothing) inside each region window.

    Pure function -- the heart of the exposure measurement, independent of any
    model. ``regions`` are (x1,y1,x2,y2) windows (typically from pose anchors).
    A region with no body pixels at all returns 0.0 (nothing to judge).
    """
    skin = np.isin(seg, list(skin_ids))
    cloth = np.isin(seg, list(clothing_ids))
    out: dict[str, float] = {}
    for name, (x1, y1, x2, y2) in regions.items():
        x1, y1 = max(0, x1), max(0, y1)
        s = int(skin[y1:y2, x1:x2].sum())
        c = int(cloth[y1:y2, x1:x2].sum())
        denom = s + c
        out[name] = float(s / denom) if denom > 0 else 0.0
    return out


def coverage_by_region(exposure: dict[str, float]) -> dict[str, float]:
    """Coverage is the complement of exposure: 1.0 = fully covered by clothing."""
    return {r: float(1.0 - e) for r, e in exposure.items()}


# Per-region ANATOMICAL mapping (Sapiens Goliath classes). Counting only the
# relevant body part per region avoids contamination -- e.g. hands resting near
# the thighs must not be read as "exposed thigh".
REGION_SKIN_CLASSES: dict[str, set[str]] = {
    "shoulder":  {"Left Upper Arm", "Right Upper Arm", "Torso"},
    "upper_arm": {"Left Upper Arm", "Right Upper Arm"},
    "midriff":   {"Torso"},
    "thigh":     {"Left Upper Leg", "Right Upper Leg"},
    "knee":      {"Left Upper Leg", "Right Upper Leg", "Left Lower Leg", "Right Lower Leg"},
    "calf":      {"Left Lower Leg", "Right Lower Leg"},
}
REGION_CLOTH_CLASSES: dict[str, set[str]] = {
    "shoulder":  {"Upper Clothing"},
    "upper_arm": {"Upper Clothing"},
    "midriff":   {"Upper Clothing"},
    "thigh":     {"Lower Clothing"},
    "knee":      {"Lower Clothing"},
    "calf":      {"Lower Clothing"},
}


def region_exposure_anatomical(
    seg: np.ndarray,
    label_map: dict[int, str],
    regions: dict[str, tuple[int, int, int, int]],
) -> dict[str, float]:
    """exposure = bare-part / (bare-part + covering-garment) inside each window.

    Uses only the body part anatomically relevant to each region (e.g. thigh ->
    Upper Leg vs Lower Clothing), so unrelated skin (hands, face) never inflates
    a region's exposure. The pose window still separates e.g. knee from calf.
    """
    out: dict[str, float] = {}
    for region, (x1, y1, x2, y2) in regions.items():
        skin_ids = class_ids(label_map, REGION_SKIN_CLASSES.get(region, set()))
        cloth_ids = class_ids(label_map, REGION_CLOTH_CLASSES.get(region, set()))
        x1, y1 = max(0, x1), max(0, y1)
        sub = seg[y1:y2, x1:x2]
        s = int(np.isin(sub, list(skin_ids)).sum())
        c = int(np.isin(sub, list(cloth_ids)).sum())
        out[region] = float(s / (s + c)) if (s + c) > 0 else 0.0
    return out


class SapiensParser:
    """Meta Sapiens body-part segmentation (Model 1).

    Sapiens seg checkpoints are distributed as TorchScript; pass the path to a
    ``.pt2``/``.pt`` checkpoint (e.g. ``sapiens_0.3b_goliath_best.pt2``). Sizes:
    0.3B (lightest) · 1B · 2B · Sapiens2 0.4B–5B (native 1K/4K).
    """

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        input_size: tuple[int, int] = (1024, 768),  # (H, W) Sapiens default
        label_map: dict[int, str] | None = None,
    ):
        import torch  # lazy

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(checkpoint, map_location=self.device).eval()
        self.input_size = input_size
        self.label_map = label_map or dict(enumerate(GOLIATH_CLASSES))
        self.skin_ids = class_ids(self.label_map, SKIN_CLASSES)
        self.clothing_ids = class_ids(self.label_map, CLOTHING_CLASSES)

    def parse(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return an HxW array of class ids for one person crop (RGB uint8).

        IMPORTANT: the released Sapiens TorchScript checkpoints have the
        ImageNet normalisation baked in, so they expect **raw 0-255 RGB** input.
        Normalising here (the obvious thing to do) double-normalises and makes
        the model predict 100% background -- verified empirically.
        """
        torch = self.torch
        # A BGR->RGB view (``[..., ::-1]``) has negative strides, which
        # torch.from_numpy rejects -- make it contiguous first.
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        x = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0)  # raw 0-255
        x = torch.nn.functional.interpolate(x, size=self.input_size, mode="bilinear",
                                            align_corners=False).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                                 align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy()

    def exposure_by_region(self, seg, regions) -> dict[str, float]:
        # Anatomical mapping: only count the body part relevant to each region.
        return region_exposure_anatomical(seg, self.label_map, regions)


class SegformerParser:
    """Lighter fallback: HF SegFormer clothes model.

    Uses the same skin/clothing dichotomy so downstream code is identical; less
    accurate than Sapiens but smaller and pip-installable via ``transformers``.
    """

    def __init__(self, model_name: str = "mattmdjaga/segformer_b2_clothes",
                 device: str | None = None):
        import torch  # lazy
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.label_map = {int(i): n for i, n in self.model.config.id2label.items()}
        # SegFormer-clothes labels skin parts as e.g. "Left-arm","Right-leg","Face".
        skin_like = {"face", "left-arm", "right-arm", "left-leg", "right-leg", "neck", "skin"}
        cloth_like = {"upper-clothes", "pants", "skirt", "dress", "coat", "shorts",
                      "left-shoe", "right-shoe", "scarf", "jumpsuits"}
        self.skin_ids = {i for i, n in self.label_map.items() if n.lower() in skin_like}
        self.clothing_ids = {i for i, n in self.label_map.items() if n.lower() in cloth_like}

    def parse(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                                 align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy()

    def exposure_by_region(self, seg, regions) -> dict[str, float]:
        return region_exposure_ratio(seg, self.skin_ids, self.clothing_ids, regions)


def regions_from_pose(anchors, bbox) -> dict[str, tuple[int, int, int, int]]:
    """Build region boxes (shoulder/upper_arm/midriff/thigh/knee/calf) from pose.

    Falls back to proportional bands of the person bbox when a landmark is
    missing, so exposure can still be estimated (with lower confidence).
    """
    x, y, w, h = (int(v) for v in bbox)
    sy = int(anchors.shoulder_y) if anchors.shoulder_y else y + int(0.20 * h)
    hy = int(anchors.hip_y) if anchors.hip_y else y + int(0.55 * h)
    ky = int(anchors.knee_y) if anchors.knee_y else y + int(0.78 * h)
    ay = int(anchors.ankle_y) if anchors.ankle_y else y + h
    return {
        "shoulder":  (x, sy - int(0.05 * h), x + w, sy + int(0.05 * h)),
        "upper_arm": (x, sy, x + w, hy - int(0.10 * h)),
        "midriff":   (x + int(0.25 * w), hy - int(0.12 * h), x + int(0.75 * w), hy),
        "thigh":     (x, hy, x + w, ky),
        "knee":      (x, ky - int(0.04 * h), x + w, ky + int(0.04 * h)),
        "calf":      (x, ky, x + w, ay),
    }
