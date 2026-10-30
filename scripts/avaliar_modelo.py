from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Caminho do modelo
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_cifar10.keras"

# Classes oficiais do CIFAR-10
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


def main():
    print("Carregando modelo...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Carregando conjunto de teste CIFAR-10...")
    (_, _), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    # Normalização dos pixels entre 0 e 1
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()

    print("Gerando previsões...")
    logits = model.predict(x_test, batch_size=128, verbose=1)

    # Converte os logits em probabilidades
    probabilities = tf.nn.softmax(logits, axis=1).numpy()

    # Seleciona a classe com maior probabilidade
    y_pred = np.argmax(probabilities, axis=1)

    # Calcula as métricas gerais
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n========== RESULTADOS DA AVALIAÇÃO ==========")
    print(f"Acurácia:  {accuracy:.4f}")
    print(f"Precisão:  {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n========== RELATÓRIO POR CLASSE ==========")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASSES,
            zero_division=0,
            digits=4,
        )
    )

    print("\n========== MATRIZ DE CONFUSÃO ==========")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()