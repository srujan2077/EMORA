import numpy as np
import torch
from torch.utils.data import Dataset


class MOSEIDataset(Dataset):
    """
    CMU-MOSEI dataset for Mark 1 emotion recognition.

    Modalities:

        Text   -> GloVe word vectors (300 dimensions)
        Audio  -> COVAREP features (74 dimensions)
        Vision -> FACET 4.2 features (35 dimensions)

    Output emotions:

        happy
        sad
        anger
        surprise
        disgust
        fear

    Mark 1 task:

        Multi-label emotion classification.

    Target convention:

        emotion score > 0  -> 1
        emotion score <= 0 -> 0
    """

    # Label layout in CMU-MOSEI:
    #
    # 0 -> sentiment
    # 1 -> happy
    # 2 -> sad
    # 3 -> anger
    # 4 -> surprise
    # 5 -> disgust
    # 6 -> fear

    EMOTION_INDICES = [
        1,
        2,
        3,
        4,
        5,
        6
    ]


    def __init__(
        self,
        dataset,
        segment_ids,
        audio_stride=10,
        vision_stride=5
    ):

        self.dataset = dataset

        self.segment_ids = segment_ids

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        self.text_seq = (
            dataset.computational_sequences[
                "glove_vectors"
            ].data
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.audio_seq = (
            dataset.computational_sequences[
                "COVAREP"
            ].data
        )

        # ----------------------------------------------------
        # Vision
        # ----------------------------------------------------

        self.vision_seq = (
            dataset.computational_sequences[
                "FACET 4.2"
            ].data
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.label_seq = (
            dataset.computational_sequences[
                "All Labels"
            ].data
        )

        # ----------------------------------------------------
        # Temporal downsampling
        # ----------------------------------------------------

        self.audio_stride = audio_stride
        self.vision_stride = vision_stride


    def __len__(self):

        return len(
            self.segment_ids
        )


    def __getitem__(self, index):

        segment_id = self.segment_ids[
            index
        ]


        # ====================================================
        # Load features
        # ====================================================

        text = np.asarray(
            self.text_seq[
                segment_id
            ]["features"],
            dtype=np.float32
        )

        audio = np.asarray(
            self.audio_seq[
                segment_id
            ]["features"],
            dtype=np.float32
        )

        vision = np.asarray(
            self.vision_seq[
                segment_id
            ]["features"],
            dtype=np.float32
        )

        labels = np.asarray(
            self.label_seq[
                segment_id
            ]["features"],
            dtype=np.float32
        )[0]


        # ====================================================
        # Remove invalid numerical values
        # ====================================================

        # COVAREP contains Inf values in some samples.
        #
        # Convert:
        #
        # NaN  -> 0
        # +Inf -> 0
        # -Inf -> 0

        text = np.nan_to_num(
            text,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        vision = np.nan_to_num(
            vision,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        labels = np.nan_to_num(
            labels,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        # ====================================================
        # Temporal downsampling
        # ====================================================

        # Audio has a very high temporal resolution.
        #
        # Example:
        #
        # 1781 frames
        #      ↓
        # every 10th frame
        #      ↓
        # ~179 frames

        audio = audio[
            ::self.audio_stride
        ]


        # Vision also has more timesteps than text.

        vision = vision[
            ::self.vision_stride
        ]


        # ====================================================
        # Select emotion labels
        # ====================================================

        # Remove sentiment.
        #
        # Keep:
        #
        # happy
        # sad
        # anger
        # surprise
        # disgust
        # fear

        labels = labels[
            self.EMOTION_INDICES
        ]


        # ====================================================
        # Convert emotion scores to binary targets
        # ====================================================

        # CMU-MOSEI provides emotion annotation scores.
        #
        # Mark 1 treats this as binary multi-label
        # classification:
        #
        # score > 0  -> emotion present -> 1
        # score <= 0 -> emotion absent  -> 0

        labels = (
            labels > 0
        ).astype(np.float32)


        # ====================================================
        # Convert everything to PyTorch tensors
        # ====================================================

        text = torch.from_numpy(
            text
        ).float()

        audio = torch.from_numpy(
            audio
        ).float()

        vision = torch.from_numpy(
            vision
        ).float()

        labels = torch.from_numpy(
            labels
        ).float()


        # ====================================================
        # Return sample
        # ====================================================

        return {

            "text":
                text,

            "audio":
                audio,

            "vision":
                vision,

            "labels":
                labels,

            "segment_id":
                segment_id
        }