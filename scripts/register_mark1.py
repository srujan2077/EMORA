import mlflow
import mlflow.pyfunc
import torch
import numpy as np

from src.models.multimodal_model import MultimodalEmotionModel


RUN_ID = "27b85ba426b648538a68a2852d59ae7c"
CHECKPOINT = "checkpoints/mark1_best.pt"
MODEL_NAME = "EMORA-Mark1"


class EMORAModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = MultimodalEmotionModel(
            embedding_dim=128,
            fusion_dim=256,
            num_emotions=6,
            dropout=0.2
        )

        checkpoint = torch.load(
            context.artifacts["checkpoint"],
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input):

        text = torch.tensor(
            model_input["text"],
            dtype=torch.float32,
            device=self.device
        )

        audio = torch.tensor(
            model_input["audio"],
            dtype=torch.float32,
            device=self.device
        )

        vision = torch.tensor(
            model_input["vision"],
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            logits = self.model(text, audio, vision)
            probabilities = torch.sigmoid(logits)

        return probabilities.cpu().numpy()


if __name__ == "__main__":

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    example_input = {
        "text": np.zeros((1, 10, 300), dtype=np.float32),
        "audio": np.zeros((1, 20, 74), dtype=np.float32),
        "vision": np.zeros((1, 10, 35), dtype=np.float32),
    }

    with mlflow.start_run(run_id=RUN_ID):

        model_info = mlflow.pyfunc.log_model(
            name="emora_mark1_model",
            python_model=EMORAModelWrapper(),
            artifacts={
                "checkpoint": CHECKPOINT
            },
            input_example=example_input,
            code_paths=["src"],
            pip_requirements=[
                "mlflow",
                "torch",
                "numpy"
            ],
            registered_model_name=MODEL_NAME
        )

        print("\n========================================")
        print("MODEL REGISTERED SUCCESSFULLY")
        print("========================================")
        print("Model:", MODEL_NAME)
        print("Model URI:", model_info.model_uri)
        print("Run ID:", RUN_ID)
        print("========================================")