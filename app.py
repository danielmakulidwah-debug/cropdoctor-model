import os
from io import BytesIO
import gc
from typing import Annotated

import torch
import timm
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Keep memory low for small free hosts (512 MB).
torch.set_num_threads(1)

app = FastAPI(title="CropDoctor AI Backend", version="1.0.0")

# Set ALLOWED_ORIGINS on the host to your Vercel URL, e.g.
# https://your-app.vercel.app  (comma-separated for several). Defaults to "*".
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIZE_REPO = "Julians30/maize-disease-models"
SUGARCANE_REPO = "dwililiya/sugarcane-plant-diseases-classification"

MAIZE_LABELS = [
    "healthy",
    "leaf_blight",
    "leaf_spot",
    "lethal_necrosis",
    "rust",
    "streak_virus",
    "fall_armyworm",
    "grasshopper",
    "leaf_beetle",
]

SUGARCANE_LABELS = {
    0: "bacterial_blight",
    1: "healthy",
    2: "mosaic",
    3: "red_rot",
    4: "rust",
    5: "yellow_disease",
}

_maize_model = None
_sugarcane_model = None
_sugarcane_processor = None
_maize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def unload_sugarcane_model():
    global _sugarcane_model, _sugarcane_processor
    _sugarcane_model = None
    _sugarcane_processor = None
    gc.collect()


def unload_maize_model():
    global _maize_model
    _maize_model = None
    gc.collect()


def load_maize_model():
    global _maize_model
    if _maize_model is not None:
        return _maize_model

    # Only one model is kept in memory at a time (small free hosts).
    unload_sugarcane_model()

    checkpoint = hf_hub_download(
        repo_id=MAIZE_REPO,
        filename="mobilenetv3_best.pth",
    )

    model = timm.create_model(
        "mobilenetv3_large_100",
        num_classes=len(MAIZE_LABELS),
    )

    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    _maize_model = model
    return _maize_model


def load_sugarcane_model():
    global _sugarcane_model, _sugarcane_processor

    if _sugarcane_model is not None:
        return _sugarcane_model, _sugarcane_processor

    # Only one model is kept in memory at a time (small free hosts).
    unload_maize_model()

    # This repository contains a Transformers-compatible config and
    # pytorch_model.bin with six EfficientNet classes.
    _sugarcane_processor = AutoImageProcessor.from_pretrained(SUGARCANE_REPO)
    _sugarcane_model = AutoModelForImageClassification.from_pretrained(
        SUGARCANE_REPO
    )
    _sugarcane_model.eval()
    _sugarcane_model.to(DEVICE)

    return _sugarcane_model, _sugarcane_processor


def predict_maize(image: Image.Image):
    model = load_maize_model()
    tensor = _maize_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    values, indices = torch.topk(
        probabilities,
        k=min(3, len(MAIZE_LABELS))
    )

    top_index = int(indices[0])
    results = [
        {
            "class": MAIZE_LABELS[int(index)],
            "confidence": float(value),
        }
        for value, index in zip(values, indices)
    ]

    return {
        "crop": "maize",
        "class": MAIZE_LABELS[top_index],
        "confidence": float(values[0]),
        "predictions": results,
        "model": MAIZE_REPO,
    }


def predict_sugarcane(image: Image.Image):
    model, processor = load_sugarcane_model()

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    values, indices = torch.topk(
        probabilities,
        k=min(3, len(SUGARCANE_LABELS))
    )

    top_index = int(indices[0])
    results = [
        {
            "class": SUGARCANE_LABELS.get(int(index), f"class_{int(index)}"),
            "confidence": float(value),
        }
        for value, index in zip(values, indices)
    ]

    return {
        "crop": "sugarcane",
        "class": SUGARCANE_LABELS.get(top_index, f"class_{top_index}"),
        "confidence": float(values[0]),
        "predictions": results,
        "model": SUGARCANE_REPO,
    }


@app.get("/")
def root():
    return {
        "name": "CropDoctor AI Backend",
        "status": "running",
        "device": str(DEVICE),
        "endpoint": "/predict",
        "supported_crops": ["maize", "sugarcane"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "maize_model_loaded": _maize_model is not None,
        "sugarcane_model_loaded": _sugarcane_model is not None,
    }


@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File(...)],
    crop: Annotated[str, Form(...)],
):
    crop = crop.strip().lower()

    if crop not in {"maize", "sugarcane"}:
        raise HTTPException(
            status_code=400,
            detail="crop must be 'maize' or 'sugarcane'",
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")

    # Prevent accidental huge uploads.
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        pil_image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    try:
        if crop == "maize":
            return predict_maize(pil_image)
        return predict_sugarcane(pil_image)
    except Exception as exc:
        # Avoid exposing internal stack traces to the browser.
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {type(exc).__name__}",
        ) from exc
