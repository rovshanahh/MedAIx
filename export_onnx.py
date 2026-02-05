from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[0]  
CKPT_PATH = ROOT / "models" / "cxr_chexpert14_densenet121.pt"
OUT_PATH = ROOT / "models" / "cxr_chexpert14_densenet121.onnx"

IMG_SIZE = 224
NUM_LABELS = 14

def build_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_LABELS)
    return m

def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        OUT_PATH.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    print("Saved ONNX to:", OUT_PATH)

if __name__ == "__main__":
    main()