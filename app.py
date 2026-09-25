"""
Handwritten Digit Recognizer -- Streamlit demo (PyTorch version).

WHAT: A Streamlit app where you draw a digit and see the model predict it live.
WHY:  For a project submission/demo, a working interface is far more
      convincing than a script full of printed numbers.

Setup:
    pip install streamlit streamlit-drawable-canvas pillow torch torchvision pandas

Run:
    python -m streamlit run app.py

Requires 'digit_classifier.pt' to exist (produced by digit_recognition.py).
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas


# ----------------------------------------------------------------------------
# PAGE CONFIG -- must be the very first Streamlit command in the script.
# WHAT: Sets the browser tab title/icon and page width.
# WHY:  Small touch, but it's the first thing anyone sees -- a default
#       "app.py" browser tab title looks unfinished.
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✍️",
    layout="centered",
)


# ----------------------------------------------------------------------------
# MODEL DEFINITION
# Must match the architecture used during training exactly, so PyTorch knows
# how to rebuild the network before loading the saved weights into it.
# ----------------------------------------------------------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@st.cache_resource
def get_model():
    model = DigitCNN()
    model.load_state_dict(torch.load('digit_classifier.pt', map_location='cpu'))
    model.eval()
    return model


def preprocess_like_mnist(canvas_image_data):
    """
    Convert a raw canvas drawing into something that matches MNIST's style:
    cropped to content, centered, and antialiased down to 28x28 -- this
    matters a lot for real-world accuracy on freehand drawings.
    """
    img = Image.fromarray((canvas_image_data[:, :, 0]).astype('uint8'))

    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    width, height = img.size
    if width > height:
        new_width = 20
        new_height = max(1, int(height * (20 / width)))
    else:
        new_height = 20
        new_width = max(1, int(width * (20 / height)))
    img = img.resize((new_width, new_height), Image.LANCZOS)

    final_img = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    final_img.paste(img, (paste_x, paste_y))

    final_img = final_img.filter(ImageFilter.MaxFilter(3))
    return final_img


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("✍️ Handwritten Digit Recognizer")
st.caption("A CNN trained on MNIST, running live in your browser via PyTorch + Streamlit.")
st.divider()

model = get_model()

# ----------------------------------------------------------------------------
# MAIN LAYOUT -- two columns: canvas on the left, results on the right.
# WHY: Side-by-side layout reads as a deliberate "product" rather than a
#      vertical stack of unrelated widgets.
# ----------------------------------------------------------------------------
col_draw, col_result = st.columns([1, 1], gap="large")

with col_draw:
    st.subheader("Draw a digit")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=st.session_state.get("canvas_key", "canvas_0"),
    )

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        predict_clicked = st.button("🔮 Predict", use_container_width=True, type="primary")
    with button_col2:
        if st.button("🗑️ Clear", use_container_width=True):
            # Changing the canvas widget's key forces Streamlit to create a
            # brand-new, empty canvas instead of reusing the drawn-on one.
            current = st.session_state.get("canvas_key", "canvas_0")
            n = int(current.split("_")[1]) + 1
            st.session_state["canvas_key"] = f"canvas_{n}"
            st.rerun()

with col_result:
    st.subheader("Result")

    if canvas_result.image_data is None or canvas_result.image_data[:, :, 0].max() == 0:
        st.info("Draw a digit on the left, then click **Predict**.")
    else:
        processed_img = preprocess_like_mnist(canvas_result.image_data)

        if processed_img is None:
            st.info("Draw a digit on the left, then click **Predict**.")
        else:
            img_array = np.array(processed_img).astype('float32') / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

            if predict_clicked:
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    digit = int(torch.argmax(probabilities).item())
                    confidence = float(torch.max(probabilities).item()) * 100

                st.metric(label="Predicted digit", value=str(digit))
                st.progress(min(int(confidence), 100), text=f"Confidence: {confidence:.1f}%")

                st.write("**Confidence across all digits:**")
                probs_df = pd.DataFrame({
                    "Digit": [str(i) for i in range(10)],
                    "Confidence": probabilities.numpy(),
                }).set_index("Digit")
                st.bar_chart(probs_df)
            else:
                st.info("Click **Predict** to see the model's guess.")

            with st.expander("🔍 Debug: what the model actually sees (28x28)"):
                st.image(processed_img.resize((140, 140), Image.NEAREST))
                st.caption("If this looks unrecognizable, try drawing bigger and more centered.")

st.divider()
st.caption("Built with PyTorch + Streamlit • Trained on the MNIST dataset (~99% test accuracy)")