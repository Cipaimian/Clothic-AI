"""Find the correct Sapiens preprocessing by trying variants."""
import glob
import cv2
import numpy as np
import torch

from clothic.perception.parsing import GOLIATH_CLASSES, SapiensParser

img = cv2.imread(sorted(glob.glob("examples/example_violation_sleeveless.jpg"))[0])
label_map = dict(enumerate(GOLIATH_CLASSES))
parser = SapiensParser("models/sapiens/sapiens_0.3b_goliath.pt2", device="cpu")
model = parser.model

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

rgb = np.ascontiguousarray(img[..., ::-1])
bgr = np.ascontiguousarray(img)


def base(arr):
    t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(t, size=(1024, 768), mode="bilinear", align_corners=False)


variants = {
    "imagenet_rgb (current)": (mean is not None, lambda: (base(rgb) / 255.0 - mean) / std),
    "raw255_rgb":             (True, lambda: base(rgb)),
    "div255_rgb":             (True, lambda: base(rgb) / 255.0),
    "imagenet_bgr":           (True, lambda: (base(bgr) / 255.0 - mean) / std),
    "raw255_bgr":             (True, lambda: base(bgr)),
}

for name, (_, fn) in variants.items():
    x = fn()
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    seg = out.argmax(dim=1)[0].numpy()
    ids, counts = np.unique(seg, return_counts=True)
    bg = dict(zip(ids.tolist(), counts.tolist())).get(0, 0)
    bg_pct = 100 * bg / seg.size
    top = sorted(zip(ids.tolist(), counts.tolist()), key=lambda t: -t[1])[:4]
    top_str = ", ".join(f"{label_map.get(i, i)}={100*c/seg.size:.0f}%" for i, c in top)
    print(f"{name:26s} bg={bg_pct:5.1f}%  | {top_str}")
