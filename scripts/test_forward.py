import torch
from mmsdk import mmdatasdk
import numpy as np

from src.models.multimodal_model import MultimodalEmotionModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_PATH = "C:/SRUJAN/multimodal_ai/data/processed/aligned_mark1"

DATA_FILES = {
    "glove_vectors": f"{BASE_PATH}/glove_vectors_aligned.csd",
    "COVAREP": f"{BASE_PATH}/COVAREP_aligned.csd",
    "FACET 4.2": f"{BASE_PATH}/FACET42_aligned.csd",
    "All Labels": f"{BASE_PATH}/labels_aligned.csd",
}


# --------------------------------------------------
# Load aligned dataset
# --------------------------------------------------

print("Loading aligned Mark 1 dataset...")

dataset = mmdatasdk.mmdataset(
    DATA_FILES,
    BASE_PATH
)

print("Dataset loaded.")


# --------------------------------------------------
# Select one common segment
# --------------------------------------------------

label_data = dataset.computational_sequences["All Labels"].data

segment_id = next(iter(label_data))

print("\nTesting segment:", segment_id)


# --------------------------------------------------
# Extract one segment from each modality
# --------------------------------------------------

text_data = dataset.computational_sequences[
    "glove_vectors"
].data[segment_id]["features"]

audio_data = dataset.computational_sequences[
    "COVAREP"
].data[segment_id]["features"]

vision_data = dataset.computational_sequences[
    "FACET 4.2"
].data[segment_id]["features"]


# --------------------------------------------------
# Convert NumPy → PyTorch
# --------------------------------------------------

text = torch.from_numpy(np.asarray(text_data)).float().unsqueeze(0)
audio = torch.from_numpy(np.asarray(audio_data)).float().unsqueeze(0)
vision = torch.from_numpy(np.asarray(vision_data)).float().unsqueeze(0) 


print("\nInput shapes:")
print("Text:  ", text.shape)
print("Audio: ", audio.shape)
print("Vision:", vision.shape)


# --------------------------------------------------
# Select device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = MultimodalEmotionModel()
model = model.to(device)
model.eval()


text = text.to(device)
audio = audio.to(device)
vision = vision.to(device)


# --------------------------------------------------
# Full multimodal forward pass
# --------------------------------------------------

with torch.no_grad():

    logits = model(
        text,
        audio,
        vision
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nModel output:")
print("Logits:", logits)

print("\nOutput shape:", logits.shape)

print("\nMark 1 forward pass completed successfully!")