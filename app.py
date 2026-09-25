"""
OPTIONAL BONUS (Step 11): Simple web demo for your digit classifier (PyTorch version).

WHAT: A Streamlit app where you draw a digit and see the model predict it live.
WHY:  For a project submission/demo, a working interface is far more
      convincing than a script full of printed numbers.

Setup:
    pip install streamlit streamlit-drawable-canvas pillow torch torchvision

Run:
    streamlit run app.py

Requires 'digit_classifier.pt' to exist (produced by digit_recognition.py).
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas


# Model class must match the one used during training exactly, so PyTorch
# knows how to rebuild the network before loading the saved weights into it.
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


st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and let the model guess it.")

# Load the model once (cached so it doesn't reload on every interaction)
@st.cache_resource
def get_model():
    model = DigitCNN()
    model.load_state_dict(torch.load('digit_classifier.pt', map_location='cpu'))
    model.eval()   # switch to inference mode (disables dropout, etc.)
    return model

model = get_model()

# Drawing canvas: 280x280 so the user has room to draw, we'll downscale to 28x28
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_like_mnist(canvas_image_data):
    """
    Convert a raw canvas drawing into something that matches MNIST's style
    as closely as possible. This matters a lot for accuracy -- MNIST digits
    are cropped to their content, centered, and scaled a specific way, and
    a live freehand drawing looks nothing like that by default.

    Steps:
    1. Crop to the actual drawn content (removes empty black space).
    2. Resize that cropped content to fit inside a 20x20 box, preserving
       aspect ratio (this matches how MNIST digits were originally built).
    3. Paste the result into the center of a blank 28x28 image (MNIST always
       leaves a small black border -- roughly 4px on each side).
    4. Use LANCZOS resampling (high-quality antialiasing) instead of the
       default resize, so curves and loops (like in 6, 9, 4, 8) don't turn
       jagged or break apart when shrunk down.
    """
    img = Image.fromarray((canvas_image_data[:, :, 0]).astype('uint8'))

    # Step 1: crop to the bounding box of non-black (drawn) pixels
    bbox = img.getbbox()
    if bbox is None:
        return None  # nothing drawn yet
    img = img.crop(bbox)

    # Step 2: resize so the longest side becomes 20px, keeping aspect ratio
    width, height = img.size
    if width > height:
        new_width = 20
        new_height = max(1, int(height * (20 / width)))
    else:
        new_height = 20
        new_width = max(1, int(width * (20 / height)))
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Step 3: paste centered onto a blank 28x28 black canvas
    final_img = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    final_img.paste(img, (paste_x, paste_y))

    # Step 4: slightly thicken the strokes. Shrinking a large freehand
    # drawing down to 20px can leave strokes only 1px wide, which loses
    # the loop/curve detail digits like 6, 9, 4, 8 depend on. MaxFilter
    # brightens each pixel to the max of its neighbors, which -- since our
    # strokes are white (255) on a black (0) background -- has the effect
    # of thickening the white strokes slightly.
    final_img = final_img.filter(ImageFilter.MaxFilter(3))

    return final_img


if canvas_result.image_data is not None:
    processed_img = preprocess_like_mnist(canvas_result.image_data)

    if processed_img is not None:
        # Show exactly what the model sees -- extremely useful for debugging
        st.write("What the model actually sees (28x28):")
        st.image(processed_img.resize((140, 140), Image.NEAREST))

        img_array = np.array(processed_img).astype('float32') / 255.0  # normalize 0-1

        # PyTorch expects shape: (batch, channels, height, width)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        if st.button("Predict"):
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                digit = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item() * 100

            st.write(f"### Prediction: {digit}")
            st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.write("Draw a digit above, then click Predict.")