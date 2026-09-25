# Handwritten Digit Recognizer

A machine learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset, with a live interactive web demo built in Streamlit.

Draw a digit with your mouse and the model predicts it in real time.

## Demo

<!-- Add a screenshot or GIF of your app here once you have one -->
<!-- ![App demo](demo.png) -->

## Features

- CNN built with PyTorch, trained on the MNIST dataset (60,000 training images, 10,000 test images)
- Achieves ~99% accuracy on the MNIST test set
- Interactive drawing canvas UI built with Streamlit
- MNIST-style preprocessing pipeline (cropping, centering, and antialiased resizing) for better real-world accuracy on freehand drawings
- Confusion matrix and classification report for model evaluation

## Tech Stack

- **Model:** PyTorch (CNN)
- **Data:** MNIST (via `torchvision.datasets`)
- **UI:** Streamlit + `streamlit-drawable-canvas`
- **Evaluation:** scikit-learn (confusion matrix, classification report)

## Project Structure

```
DigitRecog/
├── digit_recognition.py   # Trains the CNN and saves digit_classifier.pt
├── app.py                 # Streamlit app: draw a digit, get a live prediction
├── requirements.txt       # Python dependencies
├── digit_classifier.pt    # Saved trained model weights (generated after training)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer (this project uses PyTorch, which does not yet support every brand-new Python release — check [pytorch.org](https://pytorch.org) if you hit install issues on a very recent Python version)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1      # Windows PowerShell
   # source .venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the model

```bash
python digit_recognition.py
```

This downloads the MNIST dataset automatically (first run only), trains the CNN for 10 epochs, prints accuracy metrics and a confusion matrix, and saves the trained weights to `digit_classifier.pt`.

If `digit_classifier.pt` already exists, running this script again will skip training — delete or rename the file first if you want to retrain from scratch.

### Running the app

```bash
python -m streamlit run app.py
```

This starts a local web server (typically at `http://localhost:8501`) where you can draw a digit and see the model's prediction and confidence score.

## Model Architecture

A simple CNN:

```
Conv2d(1 → 32, kernel 3x3) → ReLU → MaxPool2d
Conv2d(32 → 64, kernel 3x3) → ReLU → MaxPool2d
Flatten
Linear(→ 64) → ReLU → Dropout(0.5)
Linear(→ 10)
```

Trained with the Adam optimizer and cross-entropy loss over 10 epochs.

## Results

- **Test accuracy:** ~99% on the MNIST test set
- See the confusion matrix and classification report printed during training for a per-digit breakdown

## Future Improvements

- Data augmentation (rotation, shifting) to improve robustness to varied handwriting styles
- Deploy publicly via Streamlit Community Cloud or Hugging Face Spaces
- Experiment with deeper architectures or batch normalization for higher accuracy

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun, Corinna Cortes, and Christopher Burges
- Built with [PyTorch](https://pytorch.org) and [Streamlit](https://streamlit.io)
