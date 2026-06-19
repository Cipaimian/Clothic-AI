"""CLIP-based attribute classifier (zero/few-shot, text-promptable).

For rare, subjective attributes (ripped, sheer, crop, "formal") supervised
heads starve for data. A frozen CLIP backbone + a tiny linear probe per
attribute gets usable accuracy from a few hundred examples, and the text-prompt
nature is itself a form of explainability. With no probe trained yet, this falls
back to **zero-shot** cosine similarity against positive/negative prompts.

Lazy-imported (``open_clip``/``torch``); the base package does not need it.
"""

from __future__ import annotations

import numpy as np

# Temperature for the zero-shot positive-vs-negative prompt softmax. The earlier
# value (0.01) was far too sharp: any margin in cosine similarity, however tiny,
# saturated the probability to ~0 or ~1, so an attribute that merely edged out its
# negative prompt fired its rule at near-certainty (false accusations on compliant
# outfits). 0.07 is the canonical CLIP contrastive temperature and yields
# proportional, calibrated probabilities.
ZERO_SHOT_TEMPERATURE = 0.07

# Each attribute is defined by contrasting text prompts. Editing this dict adds
# a new attribute with zero training - the prompt IS the specification.
ATTRIBUTE_PROMPTS: dict[str, tuple[str, str]] = {
    "sleeveless":        ("a sleeveless top, bare shoulders", "a top with sleeves"),
    "short_sleeve":      ("a short-sleeve t-shirt", "a long-sleeve shirt"),
    "long_sleeve":       ("a long-sleeve shirt", "a sleeveless top"),
    "midriff_exposed":   ("a crop top showing the midriff", "a top covering the stomach"),
    "ripped_torn":       ("torn ripped clothing with holes", "intact undamaged clothing"),
    "transparent_sheer": ("sheer see-through transparent fabric", "opaque solid fabric"),
    "formal_style":      ("formal collared business attire", "casual everyday clothing"),
    "hemline_above_knee":("shorts or a skirt above the knee", "trousers reaching the ankle"),
}


class ClipAttributeClassifier:
    """Garment attribute classifier (Model 2's attribute head).

    Default is **Marqo-FashionSigLIP** (a SigLIP-2 model fine-tuned on fashion):
    far stronger than vanilla CLIP for clothing, and its sigmoid training suits
    multi-label attributes. Override ``model_name`` to use plain OpenCLIP
    (e.g. ``ViT-B-32`` + ``pretrained="openai"``) on low-resource machines.
    """

    def __init__(self, model_name: str = "hf-hub:Marqo/marqo-fashionSigLIP",
                 pretrained: str | None = None,
                 device: str | None = None, probes: dict | None = None):
        import open_clip  # lazy
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # hf-hub models carry their own weights, so no `pretrained` tag is passed.
        if model_name.startswith("hf-hub:") or pretrained is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, device=self.device
            )
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.probes = probes or {}            # optional trained linear probes
        self._text_cache: dict[str, np.ndarray] = {}

    def _encode_text(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            torch = self.torch
            with torch.no_grad():
                tok = self.tokenizer([text]).to(self.device)
                feat = self.model.encode_text(tok)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            self._text_cache[text] = feat.cpu().numpy()[0]
        return self._text_cache[text]

    def _encode_image(self, image) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            img = self.preprocess(image).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]

    def classify(self, garment_crop, attributes: list[str] | None = None) -> dict[str, float]:
        """Return attribute -> probability (0..1) for one garment crop.

        Uses a trained linear probe when available, else zero-shot prompt
        contrast (softmax over positive vs negative similarity).
        """
        names = attributes or list(ATTRIBUTE_PROMPTS.keys())
        img_feat = self._encode_image(garment_crop)
        out: dict[str, float] = {}
        for name in names:
            if name in self.probes:
                w, b = self.probes[name]
                logit = float(np.dot(img_feat, w) + b)
                out[name] = float(1.0 / (1.0 + np.exp(-logit)))
            elif name in ATTRIBUTE_PROMPTS:
                pos, neg = ATTRIBUTE_PROMPTS[name]
                sp = float(np.dot(img_feat, self._encode_text(pos)))
                sn = float(np.dot(img_feat, self._encode_text(neg)))
                # Temperature-scaled softmax over the two prompt similarities.
                e = np.exp(np.array([sp, sn]) / ZERO_SHOT_TEMPERATURE)
                out[name] = float(e[0] / e.sum())
        return out
