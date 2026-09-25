"""
HANDWRITTEN DIGIT RECOGNITION SYSTEM (PyTorch version)
========================================================
A complete ML pipeline using the MNIST dataset and a CNN, built in PyTorch
instead of TensorFlow -- because PyTorch supports Python 3.14, avoiding the
need to install a second Python version.

Run this with:  python digit_recognition.py

Requirements:
    pip install torch torchvision scikit-learn matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report


# ----------------------------------------------------------------------------
# STEP 0: PICK A DEVICE (CPU or GPU)
# WHAT: Check if a GPU is available and use it; otherwise fall back to CPU.
# WHY:  Training is much faster on GPU, but the code should still work fine
#       on CPU-only laptops -- just slower.
# ----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ----------------------------------------------------------------------------
# STEP 1: GET THE DATA
# WHAT: Download the MNIST dataset (70,000 images of handwritten digits 0-9).
# WHY:  It's clean, pre-labeled, and small enough to train fast -- the
#       standard starting dataset for learning image classification.
# WHEN: Always first, before any model code.
# ----------------------------------------------------------------------------
print("\nSTEP 1: Loading MNIST dataset...")

# transforms.ToTensor() converts images to PyTorch tensors AND automatically
# scales pixel values from 0-255 down to 0-1 -- this does what Step 3's
# "normalize" did manually in the TensorFlow version.
transform = transforms.ToTensor()

train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"  Training+validation set size: {len(train_full)}")
print(f"  Test set size: {len(test_dataset)}")


# ----------------------------------------------------------------------------
# STEP 2: EXPLORE THE DATA
# WHAT: Look at a few sample images and their labels, and check class balance.
# WHY:  Cheap sanity check that catches mismatched labels, corrupted data,
#       or class imbalance before you waste time training on bad data.
# ----------------------------------------------------------------------------
print("\nSTEP 2: Exploring the data...")

labels = [label for _, label in train_full]
print(f"  Digit counts in training set: {np.bincount(labels)}")

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    img, label = train_full[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.suptitle("Sample training images")
plt.savefig('sample_digits.png')
print("  Saved sample images to sample_digits.png")


# ----------------------------------------------------------------------------
# STEP 3: PREPROCESSING NOTE
# WHAT: Normalization (0-255 -> 0-1) already happened automatically via
#       transforms.ToTensor() in Step 1. Shape is also already correct:
#       PyTorch expects (channels, height, width) = (1, 28, 28), which
#       ToTensor() produces directly -- no manual reshape needed.
# WHY:  PyTorch's data-loading conventions differ from TensorFlow's, so this
#       step folds into Step 1 instead of being separate.
# ----------------------------------------------------------------------------
print("\nSTEP 3: Preprocessing already handled by transforms.ToTensor()")
sample_img, _ = train_full[0]
print(f"  Sample image tensor shape: {sample_img.shape}")   # torch.Size([1, 28, 28])
print(f"  Pixel value range: {sample_img.min():.2f} to {sample_img.max():.2f}")


# ----------------------------------------------------------------------------
# STEP 4: SPLIT OFF A VALIDATION SET
# WHAT: Carve 10% out of the training data to monitor performance DURING
#       training, keeping the test set completely untouched until the end.
# WHY:  Lets you catch overfitting as it happens instead of finding out too
#       late. The test set stays "pure" for an honest final score.
# ----------------------------------------------------------------------------
print("\nSTEP 4: Creating validation split...")

val_size = int(0.1 * len(train_full))
train_size = len(train_full) - val_size
train_dataset, val_dataset = random_split(
    train_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"  Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

# DataLoaders feed data to the model in batches during training -- PyTorch's
# equivalent of Keras's automatic batching inside model.fit().
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ----------------------------------------------------------------------------
# STEP 5: BUILD THE MODEL (CNN)
# WHAT: A Convolutional Neural Network -- best suited for image data because
#       it understands spatial structure (edges, curves, shapes), unlike a
#       plain dense network which treats every pixel independently.
# WHY EACH LAYER:
#   Conv2d        - learns visual patterns (edges, curves, strokes)
#   MaxPool2d     - shrinks the image, keeps strongest signals, cuts compute
#   Flatten       - converts 2D feature maps into a 1D vector
#   Dropout       - randomly disables neurons during training to prevent
#                   overfitting (memorizing instead of generalizing)
#   Linear(-> 10) - final layer outputting a raw score for each digit
#                   (softmax is applied later, inside the loss function)
# ----------------------------------------------------------------------------
print("\nSTEP 5: Building the CNN model...")

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # After two conv+pool stages, a 28x28 image becomes 5x5 with 64 channels
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 28x28 -> 13x13
        x = self.pool(torch.relu(self.conv2(x)))   # 13x13 -> 5x5
        x = x.view(x.size(0), -1)                   # flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                              # raw scores (logits)
        return x

model = DigitCNN().to(device)
print(model)


# ----------------------------------------------------------------------------
# STEP 6: DEFINE LOSS FUNCTION AND OPTIMIZER
# WHAT: Define HOW the model learns.
# WHY:
#   loss      - measures how wrong predictions are; the model tries to
#               minimize this. CrossEntropyLoss fits integer labels +
#               multi-class classification (it applies softmax internally,
#               which is why the model's forward() returns raw scores).
#   optimizer - the algorithm that adjusts weights to reduce loss.
#               Adam is a strong, low-maintenance default.
# ----------------------------------------------------------------------------
print("\nSTEP 6: Setting up loss function and optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# ----------------------------------------------------------------------------
# STEP 7: TRAIN THE MODEL
# WHAT: Feed training data through the model repeatedly (epochs), adjusting
#       weights each time to reduce error.
# WHY:  One pass isn't enough to learn general patterns. But too many passes
#       cause overfitting -- that's why we watch validation accuracy.
# WHAT TO WATCH: if training accuracy keeps rising while validation accuracy
#       stalls/drops, that's overfitting.
# ----------------------------------------------------------------------------
print("\nSTEP 7: Training the model...")

EPOCHS = 10

def evaluate(loader):
    """Helper: run the model on a dataset without updating weights, return accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from last step
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                # backward pass (compute gradients)
        optimizer.step()               # update weights

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_acc = evaluate(val_loader)
    print(f"  Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_accuracy: {val_acc:.4f}")


# ----------------------------------------------------------------------------
# STEP 8: EVALUATE ON THE TEST SET
# WHAT: Run the model ONCE on the untouched test set for a final, honest
#       accuracy score.
# WHY:  Simulates real-world performance on data the model has never seen.
# ----------------------------------------------------------------------------
print("\nSTEP 8: Evaluating on test data...")
test_acc = evaluate(test_loader)
print(f"  Test accuracy: {test_acc:.4f}")


# ----------------------------------------------------------------------------
# STEP 9: ANALYZE ERRORS (BEYOND ACCURACY)
# WHAT: Look at WHICH digits get confused with which.
# WHY:  A single accuracy number hides useful detail -- e.g. maybe your model
#       consistently mixes up 4s and 9s. That tells you where to improve.
# ----------------------------------------------------------------------------
print("\nSTEP 9: Analyzing errors (confusion matrix)...")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("  Confusion matrix (rows=actual, cols=predicted):")
print(cm)
print("\n  Classification report:")
print(classification_report(all_labels, all_preds))


# ----------------------------------------------------------------------------
# STEP 10: SAVE THE MODEL
# WHAT: Save the trained model's weights to disk.
# WHY:  Training is expensive; you don't want to retrain every time you want
#       to make a prediction. Save once, load and reuse anywhere.
# ----------------------------------------------------------------------------
print("\nSTEP 10: Saving the model...")
torch.save(model.state_dict(), 'digit_classifier.pt')
print("  Model saved as digit_classifier.pt")

print("\nDONE. To load and use it later:")
print("  model = DigitCNN()")
print("  model.load_state_dict(torch.load('digit_classifier.pt'))")
print("  model.eval()")
print("  prediction = model(new_image)")
