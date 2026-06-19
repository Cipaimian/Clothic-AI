"""Full production perception backend: the complete hybrid stack.

person detect + track  ->  pose anchors  ->  human parsing (pixel exposure)
                       ->  garment detect ->  CLIP attributes

This is the backend the redesign targets. Exposure is **pixel-measured** here
(parsing ∩ pose-derived region windows), not attribute-inferred, so
``evidence_quality`` is no longer discounted for that reason.

Entirely lazy: importing this module is cheap; the heavy models load only on
construction. Install everything with ``pip install "clothic[perception]"`` plus
``transformers`` and ``open_clip_torch`` for parsing/attributes.
"""

from __future__ import annotations

from typing import Any

from clothic.perception.base import PerceptionBackend
from clothic.perception.exposure import evidence_quality
from clothic.perception.ultralytics_backend import DEFAULT_CLASS_MAP
from clothic.schemas import GarmentEvidence, PersonObservation


class FullBackend(PerceptionBackend):
    name = "full"

    def __init__(
        self,
        person_weights: str = "yolo11n.pt",
        pose_weights: str = "yolo11n-pose.pt",
        garment_weights: str | None = None,
        parser_type: str = "sapiens",            # "sapiens" | "segformer"
        sapiens_checkpoint: str | None = None,   # required for Sapiens (TorchScript)
        segformer_model: str = "mattmdjaga/segformer_b2_clothes",
        clip_model: str = "hf-hub:Marqo/marqo-fashionSigLIP",
        class_map: dict | None = None,
        device: str | None = None,
        conf: float = 0.35,
        use_parsing: bool = True,
        use_attributes: bool = True,
    ):
        from ultralytics import YOLO  # lazy

        self.person_model = YOLO(person_weights)
        self.pose_model_path = pose_weights
        self.garment_model = YOLO(garment_weights) if garment_weights else None
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.device = device
        self.conf = conf

        # Optional heavy heads - degrade gracefully if their deps are absent.
        self.pose = None
        self.parser = None
        self.attr = None
        from clothic.perception.pose import PoseEstimator

        self.pose = PoseEstimator(pose_weights, device=device, conf=conf)
        if use_parsing:
            # Model 1: body parsing. Sapiens preferred; SegFormer as fallback.
            from clothic.perception.parsing import SapiensParser, SegformerParser

            if parser_type == "sapiens" and sapiens_checkpoint:
                self.parser = SapiensParser(sapiens_checkpoint, device=device)
            else:
                self.parser = SegformerParser(segformer_model, device=device)
        if use_attributes:
            from clothic.perception.attributes import ClipAttributeClassifier

            self.attr = ClipAttributeClassifier(clip_model, device=device)

    def _crop(self, frame, box):
        x1, y1, x2, y2 = (int(max(0, v)) for v in box)
        return frame[y1:y2, x1:x2]

    def observe(self, frame: Any, frame_index: int = 0) -> list[PersonObservation]:
        from clothic.perception.parsing import regions_from_pose

        res = self.person_model.track(
            frame, persist=True, classes=[0], conf=self.conf,
            device=self.device, tracker="bytetrack.yaml", verbose=False,
        )[0]
        if res.boxes is None:
            return []

        observations: list[PersonObservation] = []
        for box in res.boxes:
            track_id = int(box.id) if box.id is not None else -1
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            pconf = float(box.conf)
            crop = self._crop(frame, (x1, y1, x2, y2))
            if crop.size == 0:
                continue

            anchors = self.pose.estimate(crop)

            # Garment detection + CLIP attributes per garment crop.
            slots: dict[str, GarmentEvidence] = {}
            if self.garment_model is not None:
                gres = self.garment_model.predict(crop, conf=self.conf,
                                                  device=self.device, verbose=False)[0]
                for gb in gres.boxes:
                    cls_name = gres.names[int(gb.cls)]
                    mapping = self.class_map.get(cls_name)
                    if mapping is None:
                        continue
                    slot, ctype, attrs = mapping
                    attrs = dict(attrs)
                    if self.attr is not None:
                        gx1, gy1, gx2, gy2 = (int(v) for v in gb.xyxy[0].tolist())
                        gcrop = crop[gy1:gy2, gx1:gx2]
                        if gcrop.size:
                            import numpy as np
                            from PIL import Image

                            rgb = np.ascontiguousarray(gcrop[..., ::-1])  # BGR->RGB, contiguous
                            attrs.update(self.attr.classify(Image.fromarray(rgb)))
                    gconf = float(gb.conf)
                    if slot not in slots or gconf > slots[slot].conf:
                        slots[slot] = GarmentEvidence(type=ctype, conf=gconf, attributes=attrs)

            # Garment recognition confidence (what is actually being judged).
            garment_confs = [g.conf for g in slots.values()]
            clothing_conf = sum(garment_confs) / len(garment_confs) if garment_confs else pconf

            # Pixel-measured exposure from parsing ∩ pose regions.
            if self.parser is not None:
                import numpy as np

                seg = self.parser.parse(crop[..., ::-1])  # BGR->RGB
                regions = regions_from_pose(anchors, (0, 0, crop.shape[1], crop.shape[0]))
                exposure = self.parser.exposure_by_region(seg, regions)
                # Reliability = how much of the crop the parser segmented as a
                # person (skin+clothing). A clean parse + confident garment
                # recognition is strong evidence; pose only places the windows.
                person_ids = list(self.parser.skin_ids | self.parser.clothing_ids)
                person_frac = float(np.isin(seg, person_ids).mean())
                parse_quality = min(1.0, person_frac / 0.25)
                quality = min(1.0, 0.5 * parse_quality + 0.5 * clothing_conf)
            else:
                from clothic.perception.exposure import exposure_from_attributes

                exposure = exposure_from_attributes(slots.get("upper"), slots.get("lower"))
                # Attribute-derived fallback: discounted, weighted to garment conf.
                quality = 0.85 * evidence_quality(
                    detection_conf=0.3 * pconf + 0.7 * clothing_conf
                )
            observations.append(
                PersonObservation(
                    track_id=track_id,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    upper=slots.get("upper"), lower=slots.get("lower"),
                    footwear=slots.get("footwear"),
                    exposure=exposure, evidence_quality=quality, frames_seen=1,
                )
            )
        return observations
