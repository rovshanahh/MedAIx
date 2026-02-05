from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import io
import threading
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import pydicom
from pydicom.errors import InvalidDicomError

from pydantic import BaseModel
import onnxruntime as ort


# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

CXR_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
IDX_TO_CXR_LABEL = {i: lab for i, lab in enumerate(CXR_LABELS)}
NUM_LABELS = len(CXR_LABELS)

CXR_THRESHOLDS = {
    "No Finding": 0.70,
    "Enlarged Cardiomediastinum": 0.05,
    "Cardiomegaly": 0.40,
    "Lung Opacity": 0.30,
    "Lung Lesion": 0.10,
    "Edema": 0.70,
    "Consolidation": 0.50,
    "Pneumonia": 0.50,
    "Atelectasis": 0.40,
    "Pneumothorax": 0.95,
    "Pleural Effusion": 0.45,
    "Pleural Other": 0.35,
    "Fracture": 0.05,
    "Support Devices": 0.50,
}

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="MedAIx Backend", version="1.0.0-cxr-multilabel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://medaix0.vercel.app",
    ],
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PROC_DIR = Path("processed")
PROC_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".dcm", ".jpg", ".jpeg", ".png", ".tiff"}
MAX_BYTES = 50 * 1024 * 1024  # 50MB for non-DICOM
DELETE_AFTER_SECONDS = 60

CXR_ONNX_PATH = MODEL_DIR / "cxr_chexpert14_densenet121.onnx"
_ort_sess: Optional[ort.InferenceSession] = None
_feat_mu = None
_feat_std = None

# --- Learned CXR confidence (feature-space) ---
CXR_FEAT_STATS_PATH = MODEL_DIR / "cxr_feat_stats.npz"
CXR_FEAT_DIM = 1024
CONF_ALPHA = 0.15  # controls sharpness (safe default)

def load_cxr_feat_stats():
    global _feat_mu, _feat_std
    if _feat_mu is None or _feat_std is None:
        if not CXR_FEAT_STATS_PATH.exists():
            raise RuntimeError(f"Missing feature stats: {CXR_FEAT_STATS_PATH}")
        data = np.load(str(CXR_FEAT_STATS_PATH))
        _feat_mu = data["mu"].astype(np.float32)
        _feat_std = data["std"].astype(np.float32)
    return _feat_mu, _feat_std


# -----------------------------
# Schemas (API contract)
# -----------------------------
class PreprocessOut(BaseModel):
    normalized_saved_as: str
    normalized_shape: List[int]
    normalized_dtype: str
    normalized_min: float
    normalized_max: float


class DicomOut(BaseModel):
    strict_read: bool
    rows: int
    cols: int
    shape: List[int]
    modality: str
    series_uid: str
    study_uid: str
    instance_number: int
    sop_instance_uid: str
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    body_part_examined: str = ""
    series_description: str = ""
    study_description: str = ""


class DetectionOut(BaseModel):
    region: str
    modality: str
    confidence: float


class RoutingOut(BaseModel):
    model_id: str
    reason: str


class MultiLabelPredictionItem(BaseModel):
    label: str
    score: float
    active: bool


class ExplainOut(BaseModel):
    mean: float
    std: float
    p95: float
    num_slices: Optional[int] = None


class InferenceOut(BaseModel):
    predictions: List[MultiLabelPredictionItem]
    uncertainty: float
    explain: ExplainOut


class PipelineOut(BaseModel):
    detection: DetectionOut
    routing: RoutingOut
    inference: InferenceOut


class UploadResponse(BaseModel):
    status: str
    stored_as: str
    preprocess: PreprocessOut
    dicom: Optional[DicomOut] = None
    pipeline: PipelineOut
    will_delete_in_seconds: int


class SeriesResponse(BaseModel):
    status: str
    series_uid: str
    num_slices: int
    strict_read_ratio: float
    volume_saved_as: str
    volume_shape: List[int]
    pipeline: PipelineOut
    will_delete_in_seconds: int


# -----------------------------
# Deletion (reliable)
# -----------------------------
def safe_delete(path_str: str) -> None:
    try:
        p = Path(path_str)
        if p.exists() and p.is_file():
            p.unlink()
    except Exception:
        pass


def schedule_delete(paths: List[Path], seconds: int = DELETE_AFTER_SECONDS) -> None:
    def _do():
        for p in paths:
            safe_delete(str(p))

    threading.Timer(seconds, _do).start()


# -----------------------------
# Standard image validation
# -----------------------------
def validate_standard_image_bytes(data: bytes) -> None:
    if len(data) <= 1024:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file (too small)")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file (PIL verify failed)")


# -----------------------------
# DICOM helpers
# -----------------------------
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return float(x[0])
        return float(x)
    except Exception:
        return None


def _safe_str(x) -> str:
    try:
        if x is None:
            return ""
        return str(x)
    except Exception:
        return ""


def read_dicom(path: Path) -> Tuple[Any, bool]:
    strict_read = True
    try:
        ds = pydicom.dcmread(str(path), force=False, stop_before_pixels=False)
        return ds, strict_read
    except (InvalidDicomError, Exception):
        strict_read = False
        try:
            ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=False)
            return ds, strict_read
        except (InvalidDicomError, Exception):
            raise HTTPException(status_code=400, detail=f"Invalid DICOM: cannot be read: {path.name}")


def validate_dicom_dataset(ds) -> None:
    if "PixelData" not in ds:
        raise HTTPException(status_code=400, detail="Invalid DICOM: missing PixelData")
    try:
        _ = ds.pixel_array
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid DICOM: PixelData cannot be decoded")


def dicom_meta(ds, strict_read: bool) -> dict:
    try:
        arr_shape = list(getattr(ds.pixel_array, "shape", []))
    except Exception:
        arr_shape = []

    return {
        "strict_read": strict_read,
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "shape": arr_shape,
        "modality": _safe_str(getattr(ds, "Modality", "UNKNOWN")) or "UNKNOWN",
        "series_uid": _safe_str(getattr(ds, "SeriesInstanceUID", "")),
        "study_uid": _safe_str(getattr(ds, "StudyInstanceUID", "")),
        "instance_number": int(getattr(ds, "InstanceNumber", 0) or 0),
        "sop_instance_uid": _safe_str(getattr(ds, "SOPInstanceUID", "")),
        "window_center": _safe_float(getattr(ds, "WindowCenter", None)),
        "window_width": _safe_float(getattr(ds, "WindowWidth", None)),
        "body_part_examined": _safe_str(getattr(ds, "BodyPartExamined", "")),
        "series_description": _safe_str(getattr(ds, "SeriesDescription", "")),
        "study_description": _safe_str(getattr(ds, "StudyDescription", "")),
    }


def save_upload(file: UploadFile, content: bytes) -> Path:
    timestamp = int(time.time() * 1000)
    stored_name = f"{timestamp}_{file.filename}"
    out_path = UPLOAD_DIR / stored_name
    out_path.write_bytes(content)
    return out_path


# -----------------------------
# Preprocessing
# - Output: (3, 224, 224) float32 in [0,1]
# - Matches training: Resize -> Grayscale(3) -> ToTensor()
# -----------------------------
def window_image(arr: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
    arr = arr.astype(np.float32)
    if center is not None and width is not None and width > 1:
        lo = center - (width / 2.0)
        hi = center + (width / 2.0)
        arr = np.clip(arr, lo, hi)
    else:
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip(arr, lo, hi)

    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def to_3chw_resized(arr01_hw: np.ndarray) -> np.ndarray:
    """arr01_hw: (H,W) float in [0,1]. Returns (3,224,224) float in [0,1]."""
    arr01_hw = np.clip(arr01_hw, 0.0, 1.0).astype(np.float32)
    img_u8 = (arr01_hw * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    pil = pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    pil_rgb = pil.convert("RGB")  # 3 channels
    arr = (np.array(pil_rgb).astype(np.float32) / 255.0)
    arr = np.transpose(arr, (2, 0, 1))  # (3,224,224)

    return arr


def preprocess_single(path: Path, ext: str, dicom_info: Optional[dict]) -> dict:
    if ext in {".jpg", ".jpeg", ".png", ".tiff"}:
        img = Image.open(str(path)).convert("L")
        arr = np.array(img, dtype=np.float32)
        norm_hw = window_image(arr, None, None)

    elif ext == ".dcm":
        ds, _ = read_dicom(path)
        validate_dicom_dataset(ds)
        arr = np.asarray(ds.pixel_array)
        wc = dicom_info.get("window_center") if dicom_info else _safe_float(getattr(ds, "WindowCenter", None))
        ww = dicom_info.get("window_width") if dicom_info else _safe_float(getattr(ds, "WindowWidth", None))
        norm_hw = window_image(arr, wc, ww)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type for preprocessing")

    norm_3chw = to_3chw_resized(norm_hw)

    out_name = path.stem + "_norm.npy"
    out_path = PROC_DIR / out_name
    np.save(str(out_path), norm_3chw.astype(np.float32))

    return {
        "normalized_saved_as": out_path.name,
        "normalized_shape": list(norm_3chw.shape),  # (3,224,224)
        "normalized_dtype": "float32",
        "normalized_min": float(norm_3chw.min()),
        "normalized_max": float(norm_3chw.max()),
    }


# -----------------------------
# Detection + routing (simple demo)
# -----------------------------
def detect_region_modality(preprocess_info: dict, dicom_info: Optional[dict]) -> dict:
    modality = "UNKNOWN"
    region = "unknown"
    confidence = 0.35

    if dicom_info:
        modality = (dicom_info.get("modality") or "UNKNOWN").upper()
        text = " ".join([
            (dicom_info.get("body_part_examined") or ""),
            (dicom_info.get("series_description") or ""),
            (dicom_info.get("study_description") or ""),
        ]).lower()

        if any(k in text for k in ["chest", "thorax", "lung", "cxr"]):
            region = "chest"
            confidence = 0.6
        elif any(k in text for k in ["brain", "head", "cranium"]):
            region = "brain"
            confidence = 0.6
        elif any(k in text for k in ["abdomen", "liver", "kidney"]):
            region = "abdomen"
            confidence = 0.55
        else:
            bp = (dicom_info.get("body_part_examined") or "").lower()
            if bp:
                region = bp
                confidence = 0.5

    shape = preprocess_info.get("normalized_shape", [])
    if region == "unknown" and shape:
        confidence = min(confidence, 0.4)

    return {"region": region, "modality": modality, "confidence": float(confidence)}


def route_model(detection: dict, is_series: bool) -> dict:
    modality = (detection.get("modality") or "UNKNOWN").upper()
    region = (detection.get("region") or "unknown").lower()

    if region == "chest" and modality in {"CR", "DX", "XR", "OT", "UNKNOWN"}:
        return {
            "model_id": "cxr_chexpert14_densenet121_v1",
            "reason": "chest x-ray multi-label (CheXpert14) DenseNet-121"
        }

    if modality in {"CR", "DX", "XR", "OT", "UNKNOWN"}:
        return {
            "model_id": "cxr_chexpert14_densenet121_v1",
            "reason": "fallback to CXR DenseNet-121"
        }

    if is_series:
        return {"model_id": "series_generic_v1", "reason": "fallback series route"}

    return {"model_id": "single_generic_v1", "reason": "fallback single route"}


# -----------------------------
# ONNX runtime inference
# -----------------------------
def get_ort_session() -> ort.InferenceSession:
    global _ort_sess
    if _ort_sess is None:
        if not CXR_ONNX_PATH.exists():
            raise RuntimeError(
                f"Missing model file: {CXR_ONNX_PATH}. "
                f"Put cxr_baseline.onnx into api/models/"
            )
        _ort_sess = ort.InferenceSession(
            str(CXR_ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
    return _ort_sess


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def learned_cxr_confidence(arr_3chw: np.ndarray) -> float:
    mu, std = load_cxr_feat_stats()

    # normalize exactly like inference
    x0 = (arr_3chw - IMAGENET_MEAN) / IMAGENET_STD
    x = x0[None, :, :, :].astype(np.float32)

    sess = get_ort_session()
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: x})[0].reshape(-1)

    z = (logits - mu) / std
    dist = np.linalg.norm(z) / np.sqrt(len(z))

    confidence = float(np.exp(-CONF_ALPHA * dist))
    return confidence


def onnx_inference_from_3chw(arr_3chw: np.ndarray, num_slices: Optional[int] = None) -> dict:
    """
    arr_3chw: (3,224,224) float32 in [0,1]
    Returns multi-label predictions for CheXpert14.
    """
    sess = get_ort_session()
    x0 = arr_3chw.astype(np.float32)
    x0 = (x0 - IMAGENET_MEAN) / IMAGENET_STD
    x = x0[None, :, :, :]  # (1,3,224,224)

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})

    logits = np.asarray(outputs[0]).astype(np.float32).reshape(-1)  # (14,) ideally

    if logits.shape[0] != NUM_LABELS:
        raise RuntimeError(f"Model output size mismatch: expected {NUM_LABELS}, got {logits.shape[0]}")

    probs = sigmoid(logits)

    preds = []
    for i in range(NUM_LABELS):
        lab = IDX_TO_CXR_LABEL[i]
        p = float(probs[i])
        thr = CXR_THRESHOLDS.get(lab, 0.5)

        preds.append({
            "label": lab,
            "score": p,
            "active": bool(p >= thr),
        })

    preds.sort(key=lambda d: d["score"], reverse=True)

    explain = {
        "mean": float(arr_3chw.mean()),
        "std": float(arr_3chw.std()),
        "p95": float(np.percentile(arr_3chw, 95)),
    }
    if num_slices is not None:
        explain["num_slices"] = int(num_slices)
    
    confidence = learned_cxr_confidence(arr_3chw)
    uncertainty = 1.0 - confidence

    return {
        "predictions": preds,
        "uncertainty": uncertainty,
        "explain": explain,
    }


def onnx_inference_single(norm_path: Path) -> dict:
    arr_3chw = np.load(str(norm_path)).astype(np.float32)  # (3,224,224)
    return onnx_inference_from_3chw(arr_3chw)


def onnx_inference_series(volume_path: Path) -> dict:
    vol = np.load(str(volume_path)).astype(np.float32)  # (S,3,224,224)
    mean_img = vol.mean(axis=0)                          # (3,224,224)
    return onnx_inference_from_3chw(mean_img, num_slices=int(vol.shape[0]))


# -----------------------------
# API
# -----------------------------
@app.get("/")
def root():
    return {"status": "MedAIx backend running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    if ext != ".dcm" and len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50MB for non-DICOM images)")

    out_path = save_upload(file, content)

    dicom_info = None
    if ext in {".jpg", ".jpeg", ".png", ".tiff"}:
        validate_standard_image_bytes(content)
    elif ext == ".dcm":
        ds, strict_read = read_dicom(out_path)
        validate_dicom_dataset(ds)
        dicom_info = dicom_meta(ds, strict_read)

    preprocess_info = preprocess_single(out_path, ext, dicom_info)

    raw = out_path
    norm_path = PROC_DIR / preprocess_info["normalized_saved_as"]
    background_tasks.add_task(schedule_delete, [raw, norm_path], DELETE_AFTER_SECONDS)

    detection = detect_region_modality(preprocess_info, dicom_info)
    routing = route_model(detection, is_series=False)

    # If non-DICOM, infer from chosen model
    if dicom_info is None and routing["model_id"].startswith("cxr_"):
        detection["region"] = "Chest"
        detection["modality"] = "XR"
        detection["confidence"] = 0.55

    try:
        inf = onnx_inference_single(norm_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {e}")

    payload = {
        "status": "uploaded_validated_preprocessed",
        "stored_as": out_path.name,
        "preprocess": preprocess_info,
        "dicom": dicom_info,
        "pipeline": {
            "detection": detection,
            "routing": routing,
            "inference": inf
        },
        "will_delete_in_seconds": DELETE_AFTER_SECONDS
    }
    return JSONResponse(payload)


@app.post("/upload-series", response_model=SeriesResponse)
async def upload_series(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved: List[Dict[str, Any]] = []
    saved_paths: List[Path] = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext != ".dcm":
            raise HTTPException(status_code=400, detail=f"Non-DICOM file in series upload: {f.filename}")

        content = await f.read()
        out_path = save_upload(f, content)
        saved_paths.append(out_path)

        ds, strict_read = read_dicom(out_path)
        validate_dicom_dataset(ds)
        meta = dicom_meta(ds, strict_read)
        saved.append({"path": out_path, "meta": meta})

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in saved:
        sid = item["meta"].get("series_uid") or "UNKNOWN_SERIES"
        groups.setdefault(sid, []).append(item)

    chosen_series_uid = max(groups.keys(), key=lambda k: len(groups[k]))
    series_items = groups[chosen_series_uid]
    series_items.sort(key=lambda x: (x["meta"].get("instance_number", 0), x["meta"].get("sop_instance_uid", "")))

    norms = []
    processed_paths: List[Path] = []
    for item in series_items:
        p = item["path"]
        meta = item["meta"]
        info = preprocess_single(p, ".dcm", meta)
        proc_path = PROC_DIR / info["normalized_saved_as"]
        processed_paths.append(proc_path)
        norms.append(np.load(str(proc_path)))  # (3,224,224)

    volume = np.stack(norms, axis=0).astype(np.float32)  # (S,3,224,224)
    vol_name = f"{int(time.time()*1000)}_series_norm.npy"
    vol_path = PROC_DIR / vol_name
    np.save(str(vol_path), volume)

    background_tasks.add_task(
        schedule_delete,
        saved_paths + processed_paths + [vol_path],
        DELETE_AFTER_SECONDS
    )

    first_meta = series_items[0]["meta"] if series_items else None
    detection = detect_region_modality({"normalized_shape": [3, IMG_SIZE, IMG_SIZE]}, first_meta)
    routing = route_model(detection, is_series=True)

    try:
        inf = onnx_inference_series(vol_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {e}")

    strict_ratio = sum(1 for x in series_items if x["meta"].get("strict_read") is True) / max(len(series_items), 1)

    payload = {
        "status": "series_validated_preprocessed",
        "series_uid": chosen_series_uid,
        "num_slices": len(series_items),
        "strict_read_ratio": float(strict_ratio),
        "volume_saved_as": vol_path.name,
        "volume_shape": list(volume.shape),
        "pipeline": {
            "detection": detection,
            "routing": routing,
            "inference": inf
        },
        "will_delete_in_seconds": DELETE_AFTER_SECONDS
    }
    return JSONResponse(payload)