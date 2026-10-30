
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def criar_imagem_teste():
    """Cria uma imagem PNG válida em memória."""
    imagem = Image.fromarray(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    buffer = BytesIO()
    imagem.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_health():
    with TestClient(app) as client:
        resposta = client.get("/health")

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status"] == "ok"
    assert dados["modelo_em_memoria"] is True


def test_predict_endpoint_exists():
    with TestClient(app) as client:
        resposta = client.post("/predict")

    # Sem o arquivo obrigatório, o FastAPI deve retornar 422.
    assert resposta.status_code == 422


def test_predict_with_valid_image():
    with TestClient(app) as client:
        resposta = client.post(
            "/predict",
            files={
                "file": (
                    "imagem.png",
                    criar_imagem_teste(),
                    "image/png",
                )
            },
        )

    assert resposta.status_code == 200
    dados = resposta.json()
    assert "classe" in dados
    assert "confianca" in dados
    assert isinstance(dados["classe"], str)
    assert isinstance(dados["confianca"], (int, float))
    assert 0 <= dados["confianca"] <= 100


def test_predict_with_invalid_file():
    with TestClient(app) as client:
        resposta = client.post(
            "/predict",
            files={
                "file": (
                    "arquivo.txt",
                    b"isto nao e uma imagem",
                    "text/plain",
                )
            },
        )

    assert resposta.status_code == 422
    assert resposta.json()["detail"] == (
        "O arquivo enviado deve ser uma imagem."
    )


def test_predict_with_empty_file():
    with TestClient(app) as client:
        resposta = client.post(
            "/predict",
            files={
                "file": (
                    "vazio.png",
                    b"",
                    "image/png",
                )
            },
        )

    assert resposta.status_code == 400
    assert resposta.json()["detail"] == (
        "O arquivo enviado está vazio."
    )
