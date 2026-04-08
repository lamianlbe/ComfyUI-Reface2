import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Reface2CropFace:
    """Detect faces in a flux2-klein generated image, crop the corresponding
    latent region, and upscale it to the target megapixels for identity
    refinement via a second denoise pass."""

    _detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "output_megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1},
                ),
                "zero_position_id": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "max_faces": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "REFACE_CROP_REGION", "FLOAT")
    RETURN_NAMES = ("face_latent", "crop_region", "recommended_denoise")
    FUNCTION = "process"
    CATEGORY = "Reface2"

    # ------------------------------------------------------------------
    # Face detector (lazy-loaded, shared across instances)
    # ------------------------------------------------------------------

    @classmethod
    def _get_detector(cls):
        if cls._detector is None:
            import math as _math

            # fdlite uses np.math which was removed in numpy 2.0
            if not hasattr(np, "math"):
                np.math = _math

            from fdlite import FaceDetection, FaceDetectionModel

            cls._detector = FaceDetection(
                model_type=FaceDetectionModel.BACK_CAMERA
            )
        return cls._detector

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------

    def process(
        self,
        image: torch.Tensor,
        latent: dict,
        output_megapixels: float,
        zero_position_id: bool,
        max_faces: int,
    ):
        samples = latent["samples"]  # (B, 128, lat_h, lat_w)
        B, C, lat_h, lat_w = samples.shape
        img_h, img_w = image.shape[1], image.shape[2]

        # Pixel-to-latent scale (should be ~16 for flux2)
        h_scale = img_h / lat_h
        w_scale = img_w / lat_w

        # -- empty return helper ------------------------------------------
        empty_latent = {
            "samples": torch.zeros(
                B, C, 1, 1, device=samples.device, dtype=samples.dtype
            )
        }
        empty_result = (empty_latent, None, 0.0)

        # -- face detection (CPU, via fdlite) -----------------------------
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        detector = self._get_detector()
        detections = detector(pil_img)

        if not detections:
            return empty_result

        # Gather face bboxes in *pixel* coordinates
        faces = []
        for det in detections:
            bbox = det.bbox
            # fdlite bbox: xmin, ymin, width, height (normalised 0-1)
            x = bbox.xmin * img_w
            y = bbox.ymin * img_h
            w = bbox.width * img_w
            h = bbox.height * img_h
            faces.append({"x": x, "y": y, "w": w, "h": h, "area": w * h})

        # Sort descending by area
        faces.sort(key=lambda f: f["area"], reverse=True)
        max_area = faces[0]["area"]

        # Keep faces whose area > half of the largest, up to *max_faces*
        filtered = [f for f in faces if f["area"] > max_area / 2][:max_faces]

        # Expand each face bbox by 2x (centred)
        expanded_rects = []
        for f in filtered:
            cx = f["x"] + f["w"] / 2
            cy = f["y"] + f["h"] / 2
            nw = f["w"] * 2
            nh = f["h"] * 2
            expanded_rects.append(
                (cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)
            )

        # Minimum bounding rectangle covering all expanded faces
        px1 = max(0.0, min(r[0] for r in expanded_rects))
        py1 = max(0.0, min(r[1] for r in expanded_rects))
        px2 = min(float(img_w), max(r[2] for r in expanded_rects))
        py2 = min(float(img_h), max(r[3] for r in expanded_rects))

        crop_pixel_area = (px2 - px1) * (py2 - py1)
        image_pixel_area = img_w * img_h

        # If the crop covers >50 % of the image, no optimisation needed
        if crop_pixel_area > image_pixel_area * 0.5:
            return empty_result

        # -- Convert pixel crop to latent coordinates ----------------------
        lx1 = max(0, int(px1 / w_scale))
        ly1 = max(0, int(py1 / h_scale))
        lx2 = min(lat_w, math.ceil(px2 / w_scale))
        ly2 = min(lat_h, math.ceil(py2 / h_scale))

        crop_lat_h = ly2 - ly1
        crop_lat_w = lx2 - lx1
        if crop_lat_h <= 0 or crop_lat_w <= 0:
            return empty_result

        # Crop latent
        cropped = samples[:, :, ly1:ly2, lx1:lx2]

        # -- Compute target latent size from output megapixels -------------
        aspect = crop_lat_w / crop_lat_h
        # Each latent cell covers h_scale * w_scale pixels
        latent_cell_pixels = h_scale * w_scale
        target_lat_area = output_megapixels * 1e6 / latent_cell_pixels
        target_lat_h = max(1, round(math.sqrt(target_lat_area / aspect)))
        target_lat_w = max(1, round(target_lat_h * aspect))

        # Upscale the cropped latent
        upscaled = F.interpolate(
            cropped.float(),
            size=(target_lat_h, target_lat_w),
            mode="bilinear",
            align_corners=False,
        ).to(samples.dtype)

        # -- Build output latent -------------------------------------------
        out_latent = {"samples": upscaled}
        if not zero_position_id:
            # Store the original crop offset so a downstream sampler can
            # set rope_options.shift_y / shift_x to preserve spatial context.
            out_latent["position_offset_h"] = ly1
            out_latent["position_offset_w"] = lx1

        # -- Crop region (for paste-back) ----------------------------------
        crop_region = {
            "pixel_x1": int(round(px1)),
            "pixel_y1": int(round(py1)),
            "pixel_x2": int(round(px2)),
            "pixel_y2": int(round(py2)),
            "lat_x1": lx1,
            "lat_y1": ly1,
            "lat_x2": lx2,
            "lat_y2": ly2,
            "original_img_h": img_h,
            "original_img_w": img_w,
            "original_lat_h": lat_h,
            "original_lat_w": lat_w,
            "target_lat_h": target_lat_h,
            "target_lat_w": target_lat_w,
        }

        # -- Recommended denoise -------------------------------------------
        scale_factor = math.sqrt(
            (target_lat_h * target_lat_w) / (crop_lat_h * crop_lat_w)
        )
        denoise = round(min(0.55, 0.2 + 0.05 * scale_factor), 2)

        return (out_latent, crop_region, denoise)


# -- Node registration ----------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Reface2CropFace": Reface2CropFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Reface2CropFace": "Reface2 Crop Face (Flux2)",
}
