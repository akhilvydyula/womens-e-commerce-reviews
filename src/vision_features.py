"""Vision-language features from text + optional product image descriptors."""
from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd

_COLOR_WORDS = frozenset(
    "red blue green black white pink purple yellow orange brown gray grey beige "
    "navy teal burgundy cream ivory gold silver olive tan coral mint lavender "
    "maroon charcoal rust mustard plum turquoise".split()
)
_PATTERN_WORDS = frozenset(
    "striped stripe floral print plaid dotted solid pattern graphic logo "
    "paisley checkered herringbone textured lace embroidered".split()
)
_FABRIC_WORDS = frozenset(
    "cotton silk wool polyester linen denim leather suede velvet chiffon "
    "satin jersey knit cashmere fleece spandex rayon mesh sheer lace".split()
)
_VISUAL_QUALITY = frozenset(
    "photo picture image color colour shade hue lighting bright dark vivid "
    "dull faded saturated pixelated accurate inaccurate mismatch different".split()
)
_GARMENT_ATTR = frozenset(
    "neckline sleeve hem waist pocket button zipper collar cuff pleat ruffle "
    "stretchy flowy lined unlined padded hooded cropped long short".split()
)

VISION_TEXT_FEATURE_NAMES = [
    "vis_color_mentions",
    "vis_pattern_mentions",
    "vis_fabric_mentions",
    "vis_photo_accuracy_signal",
    "vis_garment_detail_mentions",
    "vis_brightness_language",
    "vis_fit_visual_mentions",
    "vis_texture_mentions",
    "vis_stain_damage_mentions",
    "vis_multicolor_signal",
]

IMAGE_FEATURE_NAMES = [
    "img_brightness",
    "img_contrast",
    "img_saturation",
    "img_aspect_ratio",
    "img_warm_tone_ratio",
    "img_edge_density",
    "img_color_entropy",
    "img_dominant_red",
    "img_dominant_green",
    "img_dominant_blue",
    "img_has_image",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _count_tokens(tokens: list[str], lexicon: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in lexicon)


def extract_vision_text_features(title: pd.Series, review: pd.Series) -> pd.DataFrame:
    combined = (title.fillna("") + " " + review.fillna("")).astype(str)
    rows = []
    for text in combined:
        tokens = _tokenize(text)
        low = text.lower()
        photo_neg = sum(
            1
            for p in (
                "different than",
                "not as pictured",
                "not as shown",
                "color off",
                "wrong color",
                "misleading photo",
                "looks different",
            )
            if p in low
        )
        photo_pos = sum(1 for p in ("as pictured", "true to color", "matches photo") if p in low)
        rows.append(
            {
                "vis_color_mentions": _count_tokens(tokens, _COLOR_WORDS),
                "vis_pattern_mentions": _count_tokens(tokens, _PATTERN_WORDS),
                "vis_fabric_mentions": _count_tokens(tokens, _FABRIC_WORDS),
                "vis_photo_accuracy_signal": photo_pos - photo_neg,
                "vis_garment_detail_mentions": _count_tokens(tokens, _GARMENT_ATTR),
                "vis_brightness_language": sum(
                    1 for w in ("bright", "dark", "vivid", "dull", "faded") if w in tokens
                ),
                "vis_fit_visual_mentions": sum(
                    1
                    for w in ("oversized", "cropped", "long", "short", "boxy", "flattering")
                    if w in tokens
                ),
                "vis_texture_mentions": sum(
                    1 for w in ("soft", "rough", "silky", "scratchy", "smooth", "coarse")
                    if w in tokens
                ),
                "vis_stain_damage_mentions": sum(
                    1 for w in ("stain", "tear", "hole", "pilling", "snag", "defect") if w in tokens
                ),
                "vis_multicolor_signal": int(
                    _count_tokens(tokens, _COLOR_WORDS) >= 2
                ),
            }
        )
    return pd.DataFrame(rows, columns=VISION_TEXT_FEATURE_NAMES)


def extract_image_features(image_bytes: bytes | None) -> dict[str, float]:
    """Low-level CV descriptors from product photo (PIL + numpy)."""
    zeros = {name: 0.0 for name in IMAGE_FEATURE_NAMES}
    zeros["img_has_image"] = 0.0
    if not image_bytes:
        return zeros

    try:
        from PIL import Image
    except ImportError:
        zeros["img_has_image"] = 1.0
        return zeros

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Sobel-like edge magnitude without scipy
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge_density = float(np.mean(gx + gy))

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2)))

    hist, _ = np.histogram(gray.flatten(), bins=32, range=(0, 1), density=True)
    hist = hist + 1e-12
    entropy = float(-np.sum(hist * np.log2(hist)))

    warm = float(np.mean(r) / (np.mean(b) + 1e-6))

    return {
        "img_brightness": brightness,
        "img_contrast": contrast,
        "img_saturation": saturation,
        "img_aspect_ratio": w / max(h, 1),
        "img_warm_tone_ratio": warm,
        "img_edge_density": edge_density,
        "img_color_entropy": entropy,
        "img_dominant_red": float(np.mean(r)),
        "img_dominant_green": float(np.mean(g)),
        "img_dominant_blue": float(np.mean(b)),
        "img_has_image": 1.0,
    }


def image_features_to_row(image_bytes: bytes | None) -> pd.Series:
    return pd.Series(extract_image_features(image_bytes))
