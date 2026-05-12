#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_endovis_single.py

Windows 单文件脚本：
- 单样本模式：对 EndoVis 外科手术机器人分割数据集的单张/单序列 .npy 输入，
  用 OpenAI 视觉模型（默认 gpt-4o，可 --model 覆盖）进行多模态 VQA 标注，
  输出严格结构化 JSON 至脚本同目录下 outputs/answers.json。
- 批处理模式：提供 --input-dir 时，递归扫描 site*/(train|test)/(image|mask)/*.npy，
  生成 manifest 并批量处理，支持暂停/恢复、分片 JSONL 输出。

本版本特点：
- 只启用 9 个问题（q1–q9），重新连续编号：
    q1: primary surgical context
    q2: anatomy/region
    q3: view
    q4: bleeding/smoke/occlusion triplet
    q5: visual quality
    q6: specular/glare
    q7: contrast normal
    q8: center occluded
    q9: smoke regions
- q1/q2 使用严格受控词表（3-class / 4-class），禁止 Unknown，并在后处理阶段强制归一化。
- 保留原有批处理功能：manifest/state/shard/metrics。
- 额外输出 global taxonomy JSONL（global-annotations-*.jsonl），只做后处理映射，不额外调用模型。
- 本次更新：在 OpenAI 视觉输入中，对所有 image_url 显式指定 detail="low"。
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Set
from collections import deque

import numpy as np
from PIL import Image
from tqdm import tqdm

# OpenAI SDK (>=1.0.0)
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, BadRequestError, AuthenticationError

# -----------------------------
# Config & constants
# -----------------------------

DEFAULT_MODEL = "gpt-4o"  # supports vision
VISION_MODEL_ALLOWLIST = {"gpt-4o", "gpt-4o-mini"}

SYSTEM_PROMPT = """
You are a professional annotator in robotic minimally invasive surgery (RMIS). You must answer a fixed set of 9 VQA questions (q1–q9) for a combined EndoVis 2017 + EndoVis 2018 dataset.

Your job is NOT free-form speculation. You must:
1) rely only on visible evidence from the provided image and mask,
2) use the controlled vocabularies below,
3) return strict JSON with an answer and a 0–1 confidence per question,
4) output in English.

========
Dataset priors (for vocabulary standardization only; do NOT hallucinate):
- All images come from EndoVis 2017 or EndoVis 2018. They are porcine (pig) robotic minimally invasive surgery (RMIS) frames, viewed through a robotic endoscopic camera.
- EndoVis 2017: porcine robot-assisted nephrectomy procedures (kidney resection–related surgery). Instruments are da Vinci® Xi tools; anatomy is focused around the kidney region, but each single frame may or may not show clear nephrectomy activity.
- EndoVis 2018: porcine training procedures with semantic segmentation annotations. Anatomy labels include kidney parenchyma, covered kidney, and small intestine. The scene is heavily focused around renal/kidney regions and adjacent abdominal organs, but not every frame is definitively a nephrectomy.

Use these priors ONLY to normalize wording and to understand the general surgical context. They do NOT allow you to invent case-specific facts.
You must still base every answer on THIS SINGLE FRAME (image + mask). If you are visually uncertain, you must express that by lowering the confidence — but for q1 and q2 you STILL must choose one of the allowed tokens (no “Unknown” for q1/q2).

========
Visual pattern hints (to help, not to hallucinate):

These cues are approximate and only for aiding interpretation. They NEVER override the actual pixels.

- kidney (parenchyma):
  • Solid organ tissue with relatively dense / fine parenchymal texture.
  • Capsule boundary may be seen; surface relatively homogeneous but with subtle fine patterns.
  • Small blood vessels appear more diffuse over the parenchyma; the organ feels “solid”.

- covered kidney (fat/fascia-covered kidney):
  • Yellowish / pale-yellow fat or fascia partially or fully covering the kidney.
  • You may still see the overall kidney contour / bulging shape beneath a thin fat/fascia layer.
  • The covering tissue is more lobulated or fluffy (fat) or thin fibrous (fascia), softening kidney edges.

- other abdominal soft tissue (peritoneum/mesentery/fat):
  • Thin membranous or sheet-like reflective surfaces (peritoneum).
  • Mesentery often appears as thin membranes with branching vessels (vascular arcades).
  • Fat appears more yellow and loose/lobulated; not as compact as solid kidney parenchyma.
  • Overall different from solid-organ kidney surface or intestinal loops.

- small intestine:
  • Tubular loops with circular or longitudinal folds/pleats (mucosal folds).
  • Lumen and tubular curvature may be visible, with thin wall and shape changes due to content.
  • Surface is thinner/shinier compared to solid organs, often forming multiple loops in the field.

Use these patterns as gentle hints, NOT as hard rules. For borderline cases, keep confidence modest (e.g. 0.4–0.7).

========
Controlled vocabularies (use EXACT tokens):

[Imaging modality/view — q3]
- "endoscopic", "robotic camera", "synthetic", "Unknown"

[Scene conditions (boolean or categorical)]
- bleeding: "yes" | "no" | "Unknown"
- smoke: "yes" | "no" | "Unknown"
- occlusion (q4.occlusion): "yes" | "no" | "Unknown"   # significant view-blocking occlusion anywhere in the frame
- center occluded by instruments (q8): "yes" | "no" | "Unknown"  # instrument overlaps central ~25% area
- specular/glare: "yes" | "no" | "Unknown"
- exposure issue (too bright/dark or over/underexposed): "yes" | "no" | "Unknown"
- blur present: "yes" | "no" | "Unknown"
- contrast normal: "yes" | "no" | "Unknown"
- background cleanliness: "clean" | "cluttered" | "Unknown"
- overall visual quality: "clear" | "blurry" | "reflective" | "mixed" | "Unknown"

[q1 Surgery context (STRICT 3-way; q1 must NEVER be Unknown)]
- "nephrectomy (robot-assisted)"
- "renal/kidney-focused procedure"
- "other abdominal procedure"

Rules for q1:
- Choose "nephrectomy (robot-assisted)" ONLY when there is strong visible evidence of a nephrectomy / kidney resection stage (e.g., kidney being clearly mobilized or resected, hilar dissection with very clear resection context, vessels/clamps strongly indicating nephrectomy).
- If the frame clearly involves kidney or covered kidney or very obvious renal-region manipulation, but you cannot confidently say it is nephrectomy, choose "renal/kidney-focused procedure".
- If the main visible anatomy is small intestine / peritoneum / mesentery / fat or other non-kidney abdominal tissues and there is no clear kidney-related context, choose "other abdominal procedure".
- Even when unsure, you MUST pick one of the three tokens. Express uncertainty via a lower confidence value rather than using any “Unknown” label for q1.

[q2 Anatomy/region (STRICT 4-way; q2 must NEVER be Unknown)]
- "small intestine"
- "kidney (parenchyma)"
- "covered kidney (fat/fascia-covered kidney)"
- "other abdominal soft tissue (peritoneum/mesentery/fat)"

Rules for q2:
- If you clearly see kidney surface tissue → "kidney (parenchyma)".
- If the kidney is largely covered by fat/fascia but the region is still identifiable as kidney → "covered kidney (fat/fascia-covered kidney)".
- If the dominant visible structure is intestinal loops with folds and tubular morphology → "small intestine".
- If the frame is dominated by peritoneum/mesentery/fat or other non-kidney abdominal soft tissue → "other abdominal soft tissue (peritoneum/mesentery/fat)".
- Even when unsure, select the most reasonable of these four with an appropriately low confidence (do NOT output "Unknown" for q2).

[Likely instruments/devices in scene (for background understanding only)]
You do NOT need to output instruments explicitly, but you may use these concepts internally:
- Instruments: "Large Needle Driver", "Prograsp Forceps", "Monopolar Curved Scissors", "Bipolar Forceps", "Vessel Sealer", "Grasping Retractor"
- Other devices: "ultrasound probe", "needle", "thread", "suction–irrigation", "clips"
- If not clearly visible, treat instruments as unknown in your reasoning.

[Scene consistency rules]
- Base all answers on THIS SINGLE FRAME (image + mask). No temporal or case-level assumptions.
- If a cue is ambiguous/small/off-axis, answer "Unknown" with low confidence for yes/no questions, but remember:
  • q1 must be one of the 3 allowed tokens (no Unknown).
  • q2 must be one of the 4 allowed tokens (no Unknown).

Definitions of visual artifacts:
- Specular/glare: sharp saturated highlights and mirror-like reflections.
- Blur: degraded edge acuity across meaningful regions (beyond shallow depth-of-field bokeh).
- Exposure issue: large-area clipping (overexposed whites) or crushed shadows; or globally too-bright/too-dark.
- Smoke: whitish haze with reduced contrast (often near electrocautery).
- Bleeding: visible pooling/streaks or active oozing.
- Occlusion (q4.occlusion): mark "yes" ONLY if the view is significantly blocked (e.g., large instrument/organ covering the camera view, heavy smear/blood, dense fog) such that key anatomy is hard to see. If an instrument is merely present but does NOT meaningfully block the view, choose "no". If unsure, choose "Unknown" with low confidence.
- Occlusion at center (q8): decide if the instrument mask overlaps the central ~25% area. If unclear, "Unknown".

========
Output format (STRICT JSON; no extra text):

{
  "answers": {
    "q1":  { "value": <string: one of 3 surgery-context tokens>, "confidence": <float 0..1> },
    "q2":  { "value": <string: one of 4 anatomy tokens>, "confidence": <float> },
    "q3":  { "value": <"endoscopic"|"robotic camera"|"synthetic"|"Unknown">, "confidence": <float> },
    "q4":  { "value": { "bleeding": <yes/no/Unknown>, "smoke": <yes/no/Unknown>, "occlusion": <yes/no/Unknown> }, "confidence": <float> },
    "q5":  { "value": <"clear"|"blurry"|"reflective"|"mixed"|"Unknown">, "confidence": <float> },
    "q6":  { "value": <yes/no/Unknown>, "confidence": <float> },
    "q7":  { "value": <yes/no/Unknown>, "confidence": <float> },
    "q8":  { "value": <yes/no/Unknown>, "confidence": <float> },
    "q9":  { "value": <string: region(s) with smoke, or "none", or "Unknown">, "confidence": <float> }
  }
}

Constraints:
- Use ONLY the controlled vocabulary where specified.
- For q1 and q2 you MUST use the new tokens above (no “Unknown”).
- For q9, if q4.smoke == "no", return "none" with high confidence.
- Confidence guideline: 0.9–1.0 if cues are large/unambiguous; 0.6–0.8 if moderate; 0.1–0.4 if faint/ambiguous; 0.0 for effectively pure guess.
- Keep answers concise; no extra sentences. Do not invent patient/case-specific facts.
IMPORTANT: Use the GT mask only to localize instruments; for all other questions (q1–q7, q9) rely primarily on the original image.


========
Questions (q1–q9):
q1  What is the primary surgical context in this frame?
q2  Which organ or anatomical region does this image belong to?
q3  What is the imaging modality/view? (endoscopic / robotic camera / synthetic)
q4  Is there bleeding, smoke, or occlusion visible in the image?  (occlusion here means significant view-blocking anywhere, not just any instrument present)
q5  How is the visual quality of the current scene? (clear / blurry / reflective / mixed)
q6  Is there specular reflection/glare?
q7  Is the image contrast normal?
q8  Is the center of the frame occluded by instruments?  (strictly the central ~25% area overlapped by instruments)
q9  Which regions contain smoke? (e.g., "upper-left", "near instrument jaws", "none", or "Unknown")


Return the strict JSON only.
""".strip()

# 9 个英文问题（q1..q9，顺序固定）
QUESTIONS_EN: List[str] = [
    "What is the primary surgical context in this frame?",  # q1
    "Which organ or anatomical region does this image belong to?",  # q2
    "What is the imaging modality/view? (endoscopic / robotic camera / synthetic)",  # q3
    "Is there bleeding, smoke, or occlusion visible in the image?",  # q4
    "How is the visual quality of the current scene? (clear / blurry / reflective / mixed)",  # q5
    "Is there specular reflection/glare?",  # q6
    "Is the image contrast normal?",  # q7
    "Is the center of the frame occluded by instruments?",  # q8
    "Which regions contain smoke?",  # q9
]

# 启用问题：q1..q9（其余完全禁用）
ENABLED_Q_KEYS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]
DISABLED_Q_KEYS: List[str] = []

def _q_num(qk: str) -> int:
    try:
        return int("".join(ch for ch in qk if ch.isdigit()))
    except Exception:
        return 999

# -----------------------------
# Controlled vocabularies for q1/q2 and others
# -----------------------------

# q2: anatomy
ORGAN_CANDIDATES = {
    "small intestine",
    "kidney (parenchyma)",
    "covered kidney (fat/fascia-covered kidney)",
    "other abdominal soft tissue (peritoneum/mesentery/fat)",
}

# q1: surgery context
Q1_CANDIDATES = {
    "nephrectomy (robot-assisted)",
    "renal/kidney-focused procedure",
    "other abdominal procedure",
}

# alias 映射：兼容旧 token/近义词到新 token（全部用 lower-case key）
Q2_ALIAS_TO_CANON: Dict[str, str] = {
    "kidney parenchyma": "kidney (parenchyma)",
    "kidney": "kidney (parenchyma)",
    "covered kidney": "covered kidney (fat/fascia-covered kidney)",
    "other/unspecified tissue": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "other tissue": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "peritoneum": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "mesentery": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "fat": "other abdominal soft tissue (peritoneum/mesentery/fat)",
}

Q1_ALIAS_TO_CANON: Dict[str, str] = {
    "robot-assisted nephrectomy": "nephrectomy (robot-assisted)",
    "nephrectomy": "nephrectomy (robot-assisted)",
    "urologic surgery (unspecified)": "renal/kidney-focused procedure",
    "renal surgery": "renal/kidney-focused procedure",
    "kidney-focused procedure": "renal/kidney-focused procedure",
    "renal/kidney focused procedure": "renal/kidney-focused procedure",
    "general/abdominal surgery (unspecified)": "other abdominal procedure",
    "general/abdominal": "other abdominal procedure",
    "abdominal surgery": "other abdominal procedure",
}

QUALITY_CLASSES = {"clear", "blurry", "reflective", "mixed", "Unknown"}
YN_UNKNOWN = {"yes", "no", "Unknown"}

# 题号常量
Q_TRIPLET = 4   # q4: bleeding/smoke/occlusion
Q_ORGAN = 2     # q2: anatomy
Q_QUALITY = 5   # q5: visual quality
BINARY_QS = {6, 7, 8}  # q6/q7/q8: yes/no/Unknown

# -----------------------------
# Utilities: image & mask
# -----------------------------

def load_npy_as_rgb_middle_frame(npy_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
    raw = np.load(npy_path)
    info = {"path": npy_path, "orig_shape": tuple(raw.shape), "orig_dtype": str(raw.dtype), "normalized": False}

    arr = raw
    if arr.ndim == 4:
        T = arr.shape[0]
        mid = T // 2
        arr = arr[mid]
    elif arr.ndim == 3:
        if arr.shape[2] not in (1, 3, 4):
            T = arr.shape[0]
            mid = T // 2
            arr = arr[mid]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.ndim != 3:
        raise ValueError(f"Post-processing unexpected shape: {arr.shape}")

    H, W, C = arr.shape
    if C not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count C={C}; expected 1,3,4.")

    if np.issubdtype(arr.dtype, np.floating):
        minv = np.nanmin(arr)
        maxv = np.nanmax(arr)
        if not np.isfinite(minv) or not np.isfinite(maxv):
            arr = np.zeros_like(arr, dtype=np.float32)
            minv, maxv = 0.0, 0.0
        if maxv > minv:
            arr = (arr - minv) / (maxv - minv)
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            info["normalized"] = True
            info["norm_min"] = float(minv)
            info["norm_max"] = float(maxv)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
            info["normalized"] = True
            info["norm_min"] = float(minv)
            info["norm_max"] = float(maxv)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if C == 4:
        arr = arr[:, :, :3]
        C = 3
    if C == 1:
        arr = np.repeat(arr, 3, axis=2)

    img = Image.fromarray(arr, mode="RGB")
    info["final_size"] = img.size
    info["final_mode"] = "RGB"
    return img, info


def load_npy_mask(mask_path: str, target_size: Tuple[int, int]) -> Tuple[Image.Image, Dict[str, Any]]:
    raw = np.load(mask_path)
    info = {"path": mask_path, "orig_shape": tuple(raw.shape), "orig_dtype": str(raw.dtype)}

    arr = raw
    if arr.ndim == 3:
        if arr.shape[2] != 1:
            raise ValueError(f"GT mask expected HxW or HxWx1, got shape {arr.shape}")
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError(f"GT mask expected 2D or 3D with single channel, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    arr = arr.astype(np.int64, copy=False)
    arr[arr < 0] = 0

    Ht, Wt = target_size[1], target_size[0]
    if arr.shape[0] != Ht or arr.shape[1] != Wt:
        pil_mask = Image.fromarray(arr.astype(np.int32), mode="I")
        pil_mask = pil_mask.resize((Wt, Ht), resample=Image.NEAREST)
        arr = np.array(pil_mask, dtype=np.int64)
        info["resized_to"] = (Wt, Ht)

    palette = np.array([
        [20, 20, 20], [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
    ], dtype=np.uint8)

    H, W = arr.shape
    idx = arr % len(palette)
    rgb = palette[idx]

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    info["final_size"] = img.size
    info["final_mode"] = "RGB"
    info["unique_labels"] = int(np.unique(arr).size)
    return img, info


def pil_to_data_url_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def iso_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_questions_mapping() -> Dict[str, str]:
    # 仅返回启用的问题文本（q1..q9）
    mapping: Dict[str, str] = {}
    for idx, text in enumerate(QUESTIONS_EN, start=1):
        qk = f"q{idx}"
        if qk in ENABLED_Q_KEYS:
            mapping[qk] = text
    return dict(sorted(mapping.items(), key=lambda kv: _q_num(kv[0])))


def build_final_schema(model_name: str, image_id: str, gt_id: str,
                       answers: Optional[Dict[str, Any]], error: Any = None) -> Dict[str, Any]:
    return {
        "dataset": "EndoVis2017_2018",
        "task": "surgical_vqa_annotation",
        "model": model_name,
        "image_id": image_id,
        "gt_id": gt_id,
        "created_at": iso_now_utc(),
        "questions": build_questions_mapping(),
        "answers": answers,
        "error": error,
    }


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _yn_norm(x: Any) -> str:
    s = str(x).strip()
    sl = s.lower()
    if sl in {"yes", "no"}:
        return sl
    if s in YN_UNKNOWN:
        return s
    return "Unknown"


# -----------------------------
# q1/q2 归一化工具
# -----------------------------

def _normalize_q2_value(value: Any, conf: float) -> Tuple[str, float]:
    """
    q2: Anatomy/region 强制归一化到合法 token。
    优先：canonical match（不改置信度） > alias 映射 > 回落到 other abdominal soft tissue（并 clamp conf<=0.6）。
    """
    v_raw = str(value).strip()
    v_lower = v_raw.lower()
    canonical_map = {s.lower(): s for s in ORGAN_CANDIDATES}
    alias_map = {k.lower(): v for k, v in Q2_ALIAS_TO_CANON.items()}

    # direct match
    if v_lower in canonical_map:
        return canonical_map[v_lower], conf

    # alias
    if v_lower in alias_map:
        return alias_map[v_lower], conf

    # 完全不匹配：回落
    fallback = "other abdominal soft tissue (peritoneum/mesentery/fat)"
    return fallback, min(conf, 0.6)


def _normalize_q1_value(value: Any, conf: float, q2_value: Optional[str]) -> Tuple[str, float]:
    """
    q1: Surgery context 强制归一化到合法 token。
    优先：canonical match > alias 映射 > 根据 q2 回落（kidney-like -> renal/kidney-focused, 否则 other abdominal）。
    回落时 clamp conf<=0.6。
    """
    v_raw = str(value).strip()
    v_lower = v_raw.lower()
    canonical_map = {s.lower(): s for s in Q1_CANDIDATES}
    alias_map = {k.lower(): v for k, v in Q1_ALIAS_TO_CANON.items()}

    # direct match
    if v_lower in canonical_map:
        return canonical_map[v_lower], conf

    # alias
    if v_lower in alias_map:
        return alias_map[v_lower], conf

    # 完全不匹配：根据 q2 回落
    kidney_like = {
        "kidney (parenchyma)",
        "covered kidney (fat/fascia-covered kidney)",
    }
    if q2_value in kidney_like:
        fallback = "renal/kidney-focused procedure"
    else:
        fallback = "other abdominal procedure"
    return fallback, min(conf, 0.6)


def normalize_and_validate_answers_enabled(raw: Dict[str, Any],
                                           enabled_keys: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    只检查启用的 9 个问题（q1..q9）。
    缺失任何启用键 -> (None, reason)。
    对 q1/q2 做强制归一化（禁止 Unknown 或非法 token）：
      - q2：canonical match -> alias -> 回落到 other abdominal soft tissue。
      - q1：canonical match -> alias -> 根据已归一化的 q2 回落。
    其它问题按原逻辑最小改动：
      - q4: triplet dict（bleeding/smoke/occlusion）
      - q5: QUALITY_CLASSES
      - q6/q7/q8: yes/no/Unknown
      - q3: 任意字符串（可包含 Unknown）
      - q9: 字符串；若 q4.smoke == "no"，可强制设为 "none"（轻量归一化）
    """
    if "answers" in raw and isinstance(raw["answers"], dict):
        ans = raw["answers"]
    else:
        ans = raw

    # 1) 检查所有启用键都存在
    for k in enabled_keys:
        if k not in ans:
            return None, f"Missing key '{k}' in model answers."

    out: Dict[str, Any] = {}

    # 2) 先处理 q2（解剖），得到规范化 q2
    q2_value_canon: Optional[str] = None
    if "q2" in enabled_keys:
        item_q2 = ans.get("q2", {})
        if not isinstance(item_q2, dict):
            return None, "Answer for 'q2' is not an object."
        raw_v2 = item_q2.get("value", "other abdominal soft tissue (peritoneum/mesentery/fat)")
        conf2 = _clip01(item_q2.get("confidence", 0.0))
        q2_value_canon, conf2 = _normalize_q2_value(raw_v2, conf2)
        out["q2"] = {"value": q2_value_canon, "confidence": conf2}

    # 3) 再处理 q1（术式/上下文），使用规范化后的 q2 作回落依据
    if "q1" in enabled_keys:
        item_q1 = ans.get("q1", {})
        if not isinstance(item_q1, dict):
            return None, "Answer for 'q1' is not an object."
        raw_v1 = item_q1.get("value", "other abdominal procedure")
        conf1 = _clip01(item_q1.get("confidence", 0.0))
        q1_value_canon, conf1 = _normalize_q1_value(raw_v1, conf1, q2_value_canon)
        out["q1"] = {"value": q1_value_canon, "confidence": conf1}

    # 4) 处理其余 q3..q9
    #    按题号排序，保证 q4 在 q9 之前（便于用 q4.smoke 对 q9 做轻度归一化）
    for k in sorted(enabled_keys, key=_q_num):
        if k in ("q1", "q2"):
            continue
        item = ans.get(k, {})
        if not isinstance(item, dict):
            return None, f"Answer for '{k}' is not an object."
        value = item.get("value", "Unknown")
        conf = _clip01(item.get("confidence", 0.0))

        knum = _q_num(k)

        if knum == Q_TRIPLET:
            # q4: bleeding/smoke/occlusion
            if not isinstance(value, dict):
                return None, f"'{k}.value' must be an object with bleeding/smoke/occlusion."
            v2 = {
                "bleeding": _yn_norm(value.get("bleeding", "Unknown")),
                "smoke": _yn_norm(value.get("smoke", "Unknown")),
                "occlusion": _yn_norm(value.get("occlusion", "Unknown")),
            }
            out[k] = {"value": v2, "confidence": conf}
            continue

        if knum == Q_QUALITY:
            # q5: visual quality
            v = str(value).strip()
            std_map = {s.lower(): s for s in QUALITY_CLASSES}
            out[k] = {"value": std_map.get(v.lower(), "Unknown"), "confidence": conf}
            continue

        if knum in BINARY_QS:
            # q6/q7/q8: yes/no/Unknown
            out[k] = {"value": _yn_norm(value), "confidence": conf}
            continue

        if knum == 3:
            # q3: view，宽松处理
            out[k] = {"value": str(value).strip(), "confidence": conf}
            continue

        if knum == 9:
            # q9: smoke regions，字符串；可根据 q4.smoke 轻度归一化
            v_str = str(value).strip()
            # 如果 q4 已存在且 smoke=="no"，强制为 "none"
            q4 = out.get("q4")
            if isinstance(q4, dict):
                smoke_flag = q4.get("value", {}).get("smoke", "Unknown")
                if smoke_flag == "no":
                    v_str = "none"
            out[k] = {"value": v_str, "confidence": conf}
            continue

        # 理论不会走到这里（只有 q1..q9），但保险
        if isinstance(value, (str, int, float)):
            out[k] = {"value": str(value), "confidence": conf}
        elif isinstance(value, dict):
            out[k] = {"value": value, "confidence": conf}
        else:
            out[k] = {"value": "Unknown", "confidence": conf}

    out = dict(sorted(out.items(), key=lambda kv: _q_num(kv[0])))
    return out, None


def _apply_q1_rulebackfill(ans: dict) -> dict:
    """
    Post-normalization guard/backfill for q1 using q2（新三分类体系）：

    kidney_like = {"kidney (parenchyma)", "covered kidney (fat/fascia-covered kidney)"}

    - 若 q2 in kidney_like 且 q1 == "other abdominal procedure"：
        -> 提升为 "renal/kidney-focused procedure"，confidence 至少 0.7。
    - 若 q2 NOT in kidney_like（small intestine 或 other abdominal soft tissue）且
      q1 == "nephrectomy (robot-assisted)"：
        -> 降级为 "other abdominal procedure"，并将 confidence clamp 到 <= 0.6。
    该函数只修改 q1，其它问题原样保留。
    """
    try:
        if not isinstance(ans, dict):
            return ans

        q1 = ans.get("q1", {}) or {}
        q2 = ans.get("q2", {}) or {}
        v1 = str(q1.get("value", "")).strip()
        c1 = float(q1.get("confidence", 0.0) or 0.0)
        v2 = str(q2.get("value", "")).strip()

        kidney_like = {
            "kidney (parenchyma)",
            "covered kidney (fat/fascia-covered kidney)",
        }

        # Case 1: 解剖明显是肾区，而术式被标为 other abdominal procedure -> 提升到 renal/kidney-focused
        if v2 in kidney_like and v1 == "other abdominal procedure":
            q1["value"] = "renal/kidney-focused procedure"
            q1["confidence"] = max(c1, 0.7)
            ans["q1"] = q1
            return ans

        # Case 2: 解剖并非肾区（小肠/其它软组织），却标成 nephrectomy -> 降级
        if v2 not in kidney_like and v1 == "nephrectomy (robot-assisted)":
            q1["value"] = "other abdominal procedure"
            q1["confidence"] = min(c1, 0.6)
            ans["q1"] = q1
            return ans

        ans["q1"] = q1
        return ans
    except Exception:
        return ans


# -----------------------------
# OpenAI 调用及重试
# -----------------------------

def make_backoff_delays(max_retries: int) -> List[float]:
    max_retries = max(1, int(max_retries))
    delays = []
    d = 0.5
    for _ in range(max_retries):
        delays.append(d)
        d *= 2
    return delays


def call_openai_with_retries(client: OpenAI, payload: Dict[str, Any],
                             timeout: int, max_retries: int,
                             verbose: bool = False) -> str:
    delays = make_backoff_delays(max_retries)
    last_err = None
    for attempt, delay in enumerate(delays, start=1):
        try:
            if verbose:
                print(f"[INFO] OpenAI request attempt {attempt}/{len(delays)} ...")
            resp = client.chat.completions.create(**payload, timeout=timeout)
            text = resp.choices[0].message.content
            if not isinstance(text, str):
                raise RuntimeError("Model returned non-text content.")
            return text
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            if verbose:
                print(f"[WARN] Retryable error: {repr(e)}; sleep {delay}s")
            time.sleep(delay)
            continue
        except APIError as e:
            last_err = e
            status = getattr(e, "status", None) or getattr(e, "http_status", None)
            if status and int(status) >= 500:
                if verbose:
                    print(f"[WARN] Server error {status}; sleep {delay}s")
                time.sleep(delay)
                continue
            else:
                raise
        except BadRequestError as e:
            raise
        except AuthenticationError as e:
            raise
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[WARN] Network/unknown error: {repr(e)}; sleep {delay}s")
            time.sleep(delay)
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Unknown error during OpenAI call.")


# -----------------------------
# Batch helpers (manifest/state)
# -----------------------------

def compute_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(input_dir: Path, out_dir: Path, verbose: bool = False) -> Path:
    """
    扫描目录结构：
      site*/(train|test)/(image|mask)/*.npy
    以 image/*.npy 为基准匹配 mask/*.npy。
    写入 out_dir/manifest.jsonl
    """
    manifest_path = out_dir / "manifest.jsonl"
    tasks: List[Dict[str, Any]] = []
    input_dir = input_dir.resolve()

    # site 层
    for site_dir in sorted([p for p in input_dir.iterdir()
                            if p.is_dir() and p.name.lower().startswith("site")]):
        site = site_dir.name
        for split in ("train", "test"):
            split_dir = site_dir / split
            img_dir = split_dir / "image"
            mask_dir = split_dir / "mask"
            if not img_dir.is_dir() or not mask_dir.is_dir():
                if verbose:
                    print(f"[WARN] Missing image/mask folder under {split_dir}")
                continue

            # 遍历 image 下的 .npy
            for img_path in sorted(img_dir.glob("*.npy")):
                name = img_path.stem
                if name == ".DS_Store":
                    continue
                mpath = mask_dir / f"{name}.npy"
                if not mpath.exists():
                    if verbose:
                        print(f"[WARN] Mask not found for {img_path}")
                    continue
                tasks.append({
                    "id": name,
                    "image": str(img_path.resolve()),
                    "mask": str(mpath.resolve()),
                    "site": site,
                    "split": split
                })

    # 写 manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[INFO] Manifest written: {manifest_path} (tasks={len(tasks)})")

    return manifest_path


def load_state(out_dir: Path) -> Optional[Dict[str, Any]]:
    p = out_dir / "state.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p = out_dir / "state.json"
    tmp = out_dir / "state.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def scan_completed_from_jsonl(out_dir: Path) -> Set[str]:
    completed: Set[str] = set()
    for jf in sorted(out_dir.glob("annotations-*.jsonl")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        sid = obj.get("id")
                        if isinstance(sid, str):
                            completed.add(sid)
                    except Exception:
                        continue
        except Exception:
            continue
    return completed


# -----------------------------
# Global taxonomy helper
# -----------------------------

def _sha1_of_record(record: Dict[str, Any]) -> str:
    s = json.dumps(record, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def build_global_answers_from_dataset_answers(dataset_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据 dataset-specific answers 构造 global taxonomy answers。
    当前 EndoVis 版本中，global q1/q2 直接使用相同 token 集合；
    这里仍显式应用 q1/q2 归一化作为映射层，以便未来跨数据集扩展。

    q3..q9：直接拷贝 + 轻量校验；缺失时使用安全占位。
    """
    out: Dict[str, Any] = {}

    # ---- q2 / q1：显式映射 ----
    # q2
    ds_q2 = dataset_answers.get("q2", {})
    raw_q2_val = ds_q2.get("value", "other abdominal soft tissue (peritoneum/mesentery/fat)")
    raw_q2_conf = _clip01(ds_q2.get("confidence", 0.0))
    q2_val, q2_conf = _normalize_q2_value(raw_q2_val, raw_q2_conf)
    out["q2"] = {"value": q2_val, "confidence": q2_conf}

    # q1
    ds_q1 = dataset_answers.get("q1", {})
    raw_q1_val = ds_q1.get("value", "other abdominal procedure")
    raw_q1_conf = _clip01(ds_q1.get("confidence", 0.0))
    q1_val, q1_conf = _normalize_q1_value(raw_q1_val, raw_q1_conf, q2_val)
    out["q1"] = {"value": q1_val, "confidence": q1_conf}

    # ---- 其他 q3..q9：拷贝 + 最小校验 ----
    for qk in [f"q{i}" for i in range(3, 10)]:
        item = dataset_answers.get(qk)
        knum = _q_num(qk)

        if not isinstance(item, dict):
            # 生成安全占位
            if knum == Q_TRIPLET:
                out[qk] = {
                    "value": {"bleeding": "Unknown", "smoke": "Unknown", "occlusion": "Unknown"},
                    "confidence": 0.0,
                }
            elif knum == Q_QUALITY:
                out[qk] = {"value": "Unknown", "confidence": 0.0}
            elif knum in BINARY_QS:
                out[qk] = {"value": "Unknown", "confidence": 0.0}
            elif knum == 9:
                out[qk] = {"value": "Unknown", "confidence": 0.0}
            else:
                out[qk] = {"value": "Unknown", "confidence": 0.0}
            continue

        value = item.get("value", "Unknown")
        conf = _clip01(item.get("confidence", 0.0))

        if knum == Q_TRIPLET:
            if not isinstance(value, dict):
                v2 = {"bleeding": "Unknown", "smoke": "Unknown", "occlusion": "Unknown"}
            else:
                v2 = {
                    "bleeding": _yn_norm(value.get("bleeding", "Unknown")),
                    "smoke": _yn_norm(value.get("smoke", "Unknown")),
                    "occlusion": _yn_norm(value.get("occlusion", "Unknown")),
                }
            out[qk] = {"value": v2, "confidence": conf}
            continue

        if knum == Q_QUALITY:
            v = str(value).strip()
            std_map = {s.lower(): s for s in QUALITY_CLASSES}
            out[qk] = {"value": std_map.get(v.lower(), "Unknown"), "confidence": conf}
            continue

        if knum in BINARY_QS:
            out[qk] = {"value": _yn_norm(value), "confidence": conf}
            continue

        if knum == 3:
            out[qk] = {"value": str(value).strip(), "confidence": conf}
            continue

        if knum == 9:
            # global 层也遵循 smoke==no -> "none"
            v_str = str(value).strip()
            q4 = out.get("q4") or dataset_answers.get("q4")
            if isinstance(q4, dict):
                # 先看 global 已写入的 q4，否则从 dataset 读取
                q4_val = q4.get("value", {})
                if isinstance(q4_val, dict):
                    smoke_flag = q4_val.get("smoke", "Unknown")
                    if smoke_flag == "no":
                        v_str = "none"
            out[qk] = {"value": v_str, "confidence": conf}
            continue

        # fallback
        out[qk] = {"value": str(value), "confidence": conf}

    out = dict(sorted(out.items(), key=lambda kv: _q_num(kv[0])))
    return out


# -----------------------------
# Core single-sample inference
# -----------------------------

def run_single_inference(npy_path: str, gt_path: str, model_name: str,
                         timeout: int, max_retries: int, verbose: bool = False
                         ) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    对单个样本执行完整流程，返回 (final_json, error_field)。
    final_json 为 build_final_schema 的结果（不含 id/site/split）。
    """
    # 模型校验
    if model_name not in VISION_MODEL_ALLOWLIST:
        raise RuntimeError(
            f"The specified model '{model_name}' may not support vision inputs. "
            f"Use 'gpt-4o' or 'gpt-4o-mini'."
        )

    # API Key
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # 读原图
    img, _ = load_npy_as_rgb_middle_frame(npy_path)
    data_url_img = pil_to_data_url_png(img)

    # 读 GT
    gt_error_msg: Optional[str] = None
    data_url_gt: Optional[str] = None
    try:
        W, H = img.size
        gt_rgb, _ = load_npy_mask(gt_path, target_size=(W, H))
        data_url_gt = pil_to_data_url_png(gt_rgb)
    except Exception as e:
        gt_error_msg = f"GT mask load/resize failed: {repr(e)}"

    # 指令 + 问题清单
    questions_map = build_questions_mapping()
    instruction_text = (
        "Follow the system instruction and answer ONLY the enabled questions (q1–q9). "
        "Return STRICT JSON as specified."
    )

    client = OpenAI()
    user_content: List[Dict[str, Any]] = []
    user_content.append({"type": "text", "text": "Original image"})
    # 使用 detail="low"
    user_content.append({
        "type": "image_url",
        "image_url": {"url": data_url_img, "detail": "low"}
    })
    if data_url_gt is not None:
        user_content.append({"type": "text", "text": "GT segmentation mask (pseudocolor)"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url_gt, "detail": "low"}
        })
    else:
        user_content.append({"type": "text", "text": "GT segmentation mask unavailable due to load/resize error."})

    questions_listing = "\n".join([f"{k}: {v}" for k, v in questions_map.items()])
    user_content.append({"type": "text", "text": instruction_text})
    user_content.append({"type": "text", "text": "Here are the enabled questions:\n" + questions_listing})

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    image_id = os.path.basename(npy_path)
    gt_id_str = str(Path(gt_path).name)

    error_field: Optional[str] = None
    normalized_answers: Optional[Dict[str, Any]] = None

    try:
        model_text = call_openai_with_retries(
            client, payload, timeout=timeout, max_retries=max_retries, verbose=verbose
        )
        parsed = json.loads(model_text)
        normalized_answers, parse_err = normalize_and_validate_answers_enabled(parsed, ENABLED_Q_KEYS)
        if parse_err is not None:
            error_field = f"JSON structure/validation error: {parse_err}"
            normalized_answers = None
        if parse_err is None and isinstance(normalized_answers, dict):
            normalized_answers = _apply_q1_rulebackfill(normalized_answers)
    except Exception as e:
        # API/解析失败：使用占位 answers（q1/q2 也必须为合法 token）
        base_err = f"OpenAI/Parse error: {repr(e)}"
        error_field = base_err
        unknown_placeholder: Dict[str, Any] = {}
        for qk in ENABLED_Q_KEYS:
            knum = _q_num(qk)
            if knum == 1:
                unknown_placeholder[qk] = {
                    "value": "other abdominal procedure",
                    "confidence": 0.0,
                }
            elif knum == 2:
                unknown_placeholder[qk] = {
                    "value": "other abdominal soft tissue (peritoneum/mesentery/fat)",
                    "confidence": 0.0,
                }
            elif knum == Q_TRIPLET:
                unknown_placeholder[qk] = {
                    "value": {"bleeding": "Unknown", "smoke": "Unknown", "occlusion": "Unknown"},
                    "confidence": 0.0,
                }
            else:
                unknown_placeholder[qk] = {"value": "Unknown", "confidence": 0.0}
        normalized_answers = _apply_q1_rulebackfill(unknown_placeholder)

    if gt_error_msg:
        error_field = f"{error_field} | {gt_error_msg}" if error_field else gt_error_msg

    final_json = build_final_schema(
        model_name=model_name,
        image_id=image_id,
        gt_id=gt_id_str,
        answers=normalized_answers,
        error=error_field
    )
    return final_json, error_field


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="EndoVis2017/2018 single/batch VQA annotator (OpenAI Vision, 9-question schema).")
    # 单样本
    parser.add_argument("--input", help="Path to a single .npy file (e.g., E:\\EndoVis2017\\image.npy)")
    parser.add_argument("--gt", help="Path to a GT segmentation mask .npy (e.g., E:\\EndoVis2017\\image_mask.npy)")
    # 批处理
    parser.add_argument("--input-dir", help="Dataset root directory for batch mode (contains site1/site2/site3).")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for manifest/state/annotations (default: outputs)")
    parser.add_argument("--resume", action="store_true", help="Resume from last state (default: False)")
    parser.add_argument("--shard-size", type=int, default=10000, help="Max samples per annotations shard (default: 10000)")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush+fsync frequency (default: 1 = every sample)")
    parser.add_argument("--max-samples", type=int, default=None, help="Process only first N samples (debug)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the manifest tasks (default: False)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip samples already annotated (default: True)")
    # 实时指标
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--metrics", dest="metrics", action="store_true", default=True,
                       help="Enable live throughput/ETA metrics logging (default: on)")
    group.add_argument("--no-metrics", dest="metrics", action="store_false",
                       help="Disable live metrics logging")
    parser.add_argument("--metrics-interval", type=int, default=30,
                        help="Seconds between metrics logs (default: 30)")
    parser.add_argument("--metrics-window", type=int, default=100,
                        help="Window size (#samples) for short-term averages (default: 100)")
    parser.add_argument("--eta-smooth", type=float, default=0.6,
                        help="Exponential smoothing factor (0-1) for ETA (default: 0.6)")
    # 通用
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="OpenAI vision model (default: gpt-4o). Allowed: gpt-4o, gpt-4o-mini")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Request timeout seconds (default: 60)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="Max retry attempts on 429/5xx/network errors (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare only; no API call. Show dataset stats/questions.")
    args = parser.parse_args()

    # 固定输出目录到脚本同级 outputs（单样本模式用）
    script_dir = Path(__file__).resolve().parent
    single_output_dir = script_dir / "outputs"
    single_output_dir.mkdir(parents=True, exist_ok=True)
    single_out_json_path = single_output_dir / "answers.json"
    single_out_global_json_path = single_output_dir / "answers.global.json"

    # 分流：批处理优先
    if args.input_dir:
        # ------- Batch Mode -------
        root = Path(args.input_dir).resolve()
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 构建/读取 manifest
        manifest_path = out_dir / "manifest.jsonl"
        if not manifest_path.exists():
            manifest_path = build_manifest(root, out_dir, verbose=args.verbose)
        else:
            if args.verbose:
                print(f"[INFO] Using existing manifest: {manifest_path}")

        # 2) manifest sha1
        manifest_sha1 = compute_sha1(manifest_path)

        # 3) 读取任务
        tasks: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tasks.append(obj)
                except Exception:
                    continue

        total = len(tasks)
        if args.max_samples is not None:
            tasks = tasks[: int(args.max_samples)]
        if args.shuffle:
            random.shuffle(tasks)

        # dry-run：打印总数/示例/预估分片数
        if args.dry_run:
            shards = (len(tasks) + args.shard_size - 1) // args.shard_size if args.shard_size > 0 else 1
            print("---- DRY RUN ----")
            print(f"[DATASET] total_tasks={total} (after filters={len(tasks)}) shards={shards}")
            for i in range(min(3, len(tasks))):
                t = tasks[i]
                print(f"  sample[{i}] id={t['id']} site={t['site']} split={t['split']}")
                print(f"    image={t['image']}")
                print(f"    mask ={t['mask']}")
            qmap = build_questions_mapping()
            print(f"Using {len(qmap)} enabled English questions (q1–q9).")
            for j, (k, v) in enumerate(qmap.items()):
                if j >= 9:
                    break
                print(f"  {k}: {v}")
            print("[METRICS] dry-run mode: metrics disabled.")
            return

        # 4) 恢复或去重
        state = load_state(out_dir) if args.resume else None
        completed_ids: Set[str] = set()

        if state and state.get("manifest_sha1") == manifest_sha1:
            next_index = int(state.get("next_index", 0))
            shard_index = int(state.get("shard_index", 0))
            written_files = list(state.get("written_files", []))
            if args.verbose:
                print(f"[RESUME] next_index={next_index} shard_index={shard_index}")
        else:
            completed_ids = scan_completed_from_jsonl(out_dir) if args.skip_existing else set()
            next_index = 0
            shard_index = 0
            written_files = []

        # 5) 打开当前分片（dataset-specific & global）
        def shard_path(idx: int) -> Path:
            return out_dir / f"annotations-{idx:05d}.jsonl"

        def global_shard_path(idx: int) -> Path:
            return out_dir / f"global-annotations-{idx:05d}.jsonl"

        if state and state.get("manifest_sha1") == manifest_sha1 and "shard_index" in state:
            shard_fp = open(shard_path(shard_index), "a", encoding="utf-8")
            global_shard_fp = open(global_shard_path(shard_index), "a", encoding="utf-8")
        else:
            shard_fp = open(shard_path(shard_index), "a", encoding="utf-8")
            global_shard_fp = open(global_shard_path(shard_index), "a", encoding="utf-8")

        client_model = args.model.strip()
        timeout = int(args.timeout)
        max_retries = int(args.max_retries)

        # ---- 实时指标结构（仅本次运行周期） ----
        metrics_enabled = bool(args.metrics) and not args.dry_run
        last_metrics_print_ts = time.time()
        durations_deque = deque(maxlen=max(1, int(args.metrics_window)))
        finish_ts_deque = deque(maxlen=max(1, int(args.metrics_window)))
        global_count = 0
        global_time_acc = 0.0
        ema_eta_minutes = None

        def _fmt_hms(seconds: Optional[float]) -> str:
            if seconds is None or seconds <= 0 or not (seconds < 10**9):
                return "Unknown"
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _samples_per_minute(durations: List[float]) -> float:
            if not durations:
                return 0.0
            avg = sum(durations) / len(durations)
            return 60.0 / avg if avg > 0 else 0.0

        def _throughput_time_window(finish_ts: List[float], minutes: float) -> float:
            if not finish_ts:
                return 0.0
            cutoff = time.time() - minutes * 60.0
            recent = [ts for ts in finish_ts if ts >= cutoff]
            if len(recent) < 2:
                return 0.0
            span = max(recent) - min(recent)
            if span <= 0:
                return 0.0
            return (len(recent) - 1) / (span / 60.0)

        def _estimate_eta_minutes(remaining: int, spm_candidates: List[float]) -> Tuple[Optional[float], float]:
            for v in spm_candidates:
                if v > 0:
                    return remaining / v, v
            return None, 0.0

        # 6) 主循环
        interrupted = False
        try:
            with tqdm(total=len(tasks), desc="Batch Annotating", unit="sample", initial=next_index) as pbar:
                for idx in range(next_index, len(tasks)):
                    t = tasks[idx]
                    sid = t["id"]

                    # 跳过已存在/已完成
                    if args.skip_existing and sid in completed_ids:
                        pbar.update(1)
                        continue

                    npy_path = t["image"]
                    gt_path = t["mask"]

                    t0 = time.time()
                    # 执行单样本推理
                    try:
                        final_json, _ = run_single_inference(
                            npy_path=npy_path,
                            gt_path=gt_path,
                            model_name=client_model,
                            timeout=timeout,
                            max_retries=max_retries,
                            verbose=args.verbose
                        )
                    except KeyboardInterrupt:
                        interrupted = True
                        raise
                    except Exception as e:
                        # 构造失败结果（answers=null + error）
                        final_json = build_final_schema(
                            model_name=client_model,
                            image_id=os.path.basename(npy_path),
                            gt_id=os.path.basename(gt_path),
                            answers=None,
                            error=f"Fatal error: {repr(e)}"
                        )

                    # 增加批处理字段（dataset-specific record）
                    record = {
                        "id": sid,
                        "site": t["site"],
                        "split": t["split"],
                        **final_json
                    }

                    # 生成 global record（后处理，不额外调用模型）
                    dataset_answers = final_json.get("answers") or {}
                    global_answers = build_global_answers_from_dataset_answers(dataset_answers) if dataset_answers else {}
                    # 元数据：追溯
                    metadata = {
                        "source_dataset": final_json.get("dataset", "EndoVis2017_2018"),
                        "source_id": sid,
                        "source_record_hash": _sha1_of_record(record),
                    }
                    global_record = {
                        **record,
                        "answers": global_answers if global_answers else None,
                        "metadata": metadata,
                    }

                    # 分片切换：每满 shard_size 条切换到下一个文件
                    if args.shard_size > 0:
                        current_count = (idx) % args.shard_size
                        if current_count == 0 and idx != next_index:
                            shard_fp.close()
                            global_shard_fp.close()
                            shard_index += 1
                            shard_fp = open(shard_path(shard_index), "a", encoding="utf-8")
                            global_shard_fp = open(global_shard_path(shard_index), "a", encoding="utf-8")

                    # 追加写入一行（dataset-specific）
                    shard_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    # 追加写入一行（global taxonomy）
                    global_shard_fp.write(json.dumps(global_record, ensure_ascii=False) + "\n")

                    # 刷盘策略
                    if args.flush_every <= 1 or (idx + 1) % args.flush_every == 0:
                        shard_fp.flush()
                        os.fsync(shard_fp.fileno())
                        global_shard_fp.flush()
                        os.fsync(global_shard_fp.fileno())

                        # 更新并保存进度（state 仍只记录 dataset-specific 文件）
                        state = {
                            "manifest_sha1": manifest_sha1,
                            "total_tasks": len(tasks),
                            "next_index": idx + 1,
                            "completed": idx + 1,
                            "shard_index": shard_index,
                            "written_files": [str(p) for p in sorted(out_dir.glob("annotations-*.jsonl"))],
                        }
                        save_state(out_dir, state)

                    # --- 指标更新与周期打印 ---
                    t1 = time.time()
                    duration = max(0.0, t1 - t0)
                    if metrics_enabled:
                        durations_deque.append(duration)
                        finish_ts_deque.append(t1)
                        global_count += 1
                        global_time_acc += duration

                        now_ts = time.time()
                        if (now_ts - last_metrics_print_ts) >= max(1, int(args.metrics_interval)):
                            last_metrics_print_ts = now_ts
                            spm_recent = _samples_per_minute(list(durations_deque))
                            spm_1m = _throughput_time_window(list(finish_ts_deque), 1.0)
                            spm_5m = _throughput_time_window(list(finish_ts_deque), 5.0)
                            spm_10m = _throughput_time_window(list(finish_ts_deque), 10.0)
                            spm_global = 60.0 * global_count / global_time_acc if (global_count and global_time_acc > 0) else 0.0

                            processed_so_far = idx + 1
                            total_tasks = len(tasks)
                            remaining = max(0, total_tasks - processed_so_far)
                            eta_minutes, spm_used = _estimate_eta_minutes(
                                remaining,
                                [spm_recent, spm_1m, spm_5m, spm_10m, spm_global]
                            )
                            if eta_minutes is not None:
                                a = max(0.0, min(1.0, float(args.eta_smooth)))
                                if ema_eta_minutes is None:
                                    ema_eta_minutes = eta_minutes
                                else:
                                    ema_eta_minutes = a * eta_minutes + (1 - a) * ema_eta_minutes

                            print(
                                "[METRICS] "
                                f"done={processed_so_far}/{total_tasks} | "
                                f"recent_spm={spm_recent:.2f} | "
                                f"1m={spm_1m:.2f} 5m={spm_5m:.2f} 10m={spm_10m:.2f} | "
                                f"global_spm={spm_global:.2f} | "
                                f"ETA={_fmt_hms(ema_eta_minutes*60.0 if ema_eta_minutes else None)}"
                            )

                    pbar.update(1)

        except KeyboardInterrupt:
            interrupted = True
        finally:
            try:
                shard_fp.flush()
                os.fsync(shard_fp.fileno())
                shard_fp.close()
            except Exception:
                pass
            try:
                global_shard_fp.flush()
                os.fsync(global_shard_fp.fileno())
                global_shard_fp.close()
            except Exception:
                pass

        # 7) 收尾：保存最终 state
        state = state or {}
        final_state = {
            "manifest_sha1": manifest_sha1,
            "total_tasks": len(tasks),
            "next_index": len(tasks) if not interrupted else int(state.get("next_index", 0)),
            "completed": len(tasks) if not interrupted else int(state.get("completed", 0)),
            "shard_index": shard_index,
            "written_files": [str(p) for p in sorted(out_dir.glob("annotations-*.jsonl"))],
        }
        save_state(out_dir, final_state)
        if interrupted:
            print("[INFO] Interrupted. Progress saved. You can resume with --resume.")
        else:
            print(f"[OK] Batch completed. Results under: {out_dir}")
        return

    # ------- Single-sample Mode -------
    if not args.input or not args.gt:
        print("[Error] In single-sample mode, --input and --gt are required. Or use --input-dir for batch.",
              file=sys.stderr)
        sys.exit(2)

    model_name = args.model.strip()
    timeout = int(args.timeout)
    max_retries = int(args.max_retries)

    # dry-run 单样本：打印基础信息并输出 answers=null
    if args.dry_run:
        img, img_info = load_npy_as_rgb_middle_frame(args.input)
        _ = pil_to_data_url_png(img)
        gt_error_msg = None
        gt_info = {}
        try:
            W, H = img.size
            gt_rgb, gt_info = load_npy_mask(args.gt, target_size=(W, H))
            _ = pil_to_data_url_png(gt_rgb)
        except Exception as e:
            gt_error_msg = f"GT mask load/resize failed: {repr(e)}"

        print("---- DRY RUN ----")
        print(f"[Image] path={img_info.get('path')} shape={img_info.get('orig_shape')} "
              f"dtype={img_info.get('orig_dtype')} -> RGB size={img_info.get('final_size')}")
        if img_info.get("normalized"):
            print(f"[Image] normalized from min={img_info.get('norm_min')} "
                  f"max={img_info.get('norm_max')} to [0,255] uint8")
        print(f"[GT]    path={args.gt}")
        if gt_error_msg:
            print(f"[GT]    WARNING: {gt_error_msg}")
        else:
            print(f"[GT]    shape={gt_info.get('orig_shape')} dtype={gt_info.get('orig_dtype')} "
                  f"-> RGB size={gt_info.get('final_size')} unique_labels={gt_info.get('unique_labels')}")
            if "resized_to" in gt_info:
                print(f"[GT]    resized_to={gt_info['resized_to']}")
        qmap = build_questions_mapping()
        print(f"Using {len(qmap)} enabled English questions (q1–q9).")
        for j, (k, v) in enumerate(qmap.items()):
            if j >= 9:
                break
            print(f"  {k}: {v}")

        image_id = os.path.basename(args.input)
        gt_id_str = str(Path(args.gt).name)
        final_json = build_final_schema(model_name, image_id, gt_id_str,
                                        answers=None, error=gt_error_msg)
        with open(single_out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        print(f"[OK] (dry-run) Saved to: {single_out_json_path}")
        return

    # 正式单样本推理
    try:
        final_json, error_field = run_single_inference(
            npy_path=args.input,
            gt_path=args.gt,
            model_name=model_name,
            timeout=timeout,
            max_retries=max_retries,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"[Error] {repr(e)}", file=sys.stderr)
        sys.exit(5)

    # 写 dataset-specific 单样本结果
    try:
        with open(single_out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Error] Failed to write output JSON: {e}", file=sys.stderr)
        sys.exit(6)

    # 生成并写 single-sample global 结果
    try:
        dataset_answers = final_json.get("answers") or {}
        global_answers = build_global_answers_from_dataset_answers(dataset_answers) if dataset_answers else {}
        metadata = {
            "source_dataset": final_json.get("dataset", "EndoVis2017_2018"),
            "source_id": final_json.get("image_id"),
            "source_record_hash": _sha1_of_record(final_json),
        }
        global_record = {
            **final_json,
            "answers": global_answers if global_answers else None,
            "metadata": metadata,
        }
        with open(single_out_global_json_path, "w", encoding="utf-8") as f:
            json.dump(global_record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write global taxonomy output: {e}", file=sys.stderr)

    print(f"[OK] Saved answers to: {single_out_json_path}")
    print(f"[OK] Saved global answers to: {single_out_global_json_path}")
    if final_json.get("error"):
        print(f"[WARN] Completed with note: {final_json['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
