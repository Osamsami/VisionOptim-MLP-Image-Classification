"""Streamlit demo: upload an image, get a prediction from the from-scratch
NumPy MLP trained by train.py.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st

from src.mlp import MLP, preprocess_image

MODEL_PATH = os.environ.get("VISIONOPTIM_MODEL_PATH", "models/mlp_weights.npz")

st.set_page_config(page_title="VisionOptim", page_icon="🖼️")

st.title("VisionOptim")
st.caption(
    "Image classification with a Multi-Layer Perceptron implemented from "
    "scratch in NumPy - no TensorFlow, PyTorch, or scikit-learn."
)


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None, None, None
    return MLP.load(path)


model, class_names, img_size = load_model(MODEL_PATH)

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. Train one first:\n\n"
        "```bash\npython train.py\n```"
    )
    st.stop()

st.success(f"Loaded model ({len(class_names)} classes: {', '.join(class_names)}).")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read that file as an image. Try a different file.")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    features = preprocess_image(image_bgr, img_size=img_size)
    pred_idx, probs = model.predict(features.reshape(1, -1))
    predicted_class = class_names[int(pred_idx[0])]
    confidence = float(probs[0, pred_idx[0]])

    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.1%}")

    st.bar_chart(
        {name: float(p) for name, p in zip(class_names, probs[0])},
    )
else:
    st.info("Upload an image above to get a prediction.")
