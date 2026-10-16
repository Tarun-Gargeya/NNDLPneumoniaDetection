"""Streamlit demo. Run with: streamlit run app.py"""
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import RESULTS, VARIANTS, PneumoniaNet, eval_resize, grad_cam, normalize  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model(variant: str):
    model = PneumoniaNet(pretrained=False, adversary=VARIANTS[variant][2]).to(DEV)
    model.load_state_dict(torch.load(RESULTS / variant / "model.pth", map_location=DEV))
    return model.eval()


st.set_page_config(page_title="Pneumonia triage", layout="centered")
st.title("Pediatric pneumonia triage")
st.caption("Course project prototype. Not for clinical use.")

available = [v for v in VARIANTS if (RESULTS / v / "model.pth").exists()]
if not available:
    st.error("No trained models found in results/. Run run_all.sh first.")
    st.stop()
variant = st.selectbox("Model", available, index=available.index("camguide_aug") if "camguide_aug" in available else 0)
threshold = st.slider("Decision threshold (lower = higher recall)", 0.05, 0.95, 0.5, 0.05)
upload = st.file_uploader("Chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if upload:
    gray = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    x = eval_resize(torch.from_numpy(small).float()[None, None].to(DEV) / 255)
    model = load_model(variant)
    logits, fmap, _, _ = model(normalize(x))
    prob = logits.float().softmax(1)[0, 1].item()
    pred = int(prob >= threshold)
    # Grad-CAM on the logit margin of the predicted class
    logits = logits.float()
    margin = logits[:, [pred]] - logits[:, [1 - pred]]
    cam = grad_cam(margin, fmap, torch.tensor([0], device=DEV), create_graph=False)
    cam = F.interpolate(cam[None], size=224, mode="bilinear")[0, 0].detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-6)

    # fade the overlay for low-confidence predictions
    certainty = abs(prob - threshold) / max(threshold, 1 - threshold)
    alpha = (0.15 + 0.45 * certainty) * cam[..., None]
    base = x[0, 0].cpu().numpy()
    overlay = (1 - alpha) * np.stack([base] * 3, -1) + alpha * cm.jet(cam)[..., :3]
    label = "PNEUMONIA" if pred else "NORMAL"
    st.metric("Prediction", label, f"P(pneumonia) = {prob:.3f}", delta_color="off", delta_arrow="off")
    c1, c2 = st.columns(2)
    c1.image(base, caption="Input (224×224)", clamp=True, width="stretch")
    c2.image(np.clip(overlay, 0, 1), caption=f"Evidence for {label} (Grad-CAM)", width="stretch")
