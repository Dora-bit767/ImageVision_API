from pathlib import Path

import numpy as np
import tensorflow as tf


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "modelo_cifar10.keras"
)

CLASSES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


class ClassifierService:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Carrega o modelo uma única vez para uso nas previsões."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

    def predict(self, image_array: np.ndarray) -> dict:
        """Recebe uma imagem preparada e retorna a classe prevista."""
        if self.model is None:
            raise RuntimeError("O modelo ainda não foi carregado.")

        image_array = np.asarray(image_array, dtype=np.float32)

        if image_array.shape != (32, 32, 3):
            raise ValueError(
                "A imagem deve ter formato (32, 32, 3)."
            )

        if not np.isfinite(image_array).all():
            raise ValueError("A imagem contém valores inválidos.")

        if image_array.min() < 0 or image_array.max() > 1:
            raise ValueError(
                "Os pixels devem estar normalizados entre 0 e 1."
            )

        batch = np.expand_dims(image_array, axis=0)
        logits = self.model.predict(batch, verbose=0)
        probabilities = tf.nn.softmax(logits[0]).numpy()

        class_id = int(np.argmax(probabilities))

        return {
            "classe": CLASSES[class_id],
            "confianca": round(float(probabilities[class_id]) * 100, 2),
        }


classifier_service = ClassifierService()