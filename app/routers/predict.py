
from io import BytesIO

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.schemas import PredictionResponse
from app.services import classifier_service


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna a classe prevista pelo modelo."""

    # Valida o tipo informado no upload antes de processar a imagem.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="O arquivo enviado deve ser uma imagem.",
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="O arquivo enviado está vazio.",
            )

        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize((32, 32))
            image_array = np.asarray(image, dtype=np.float32) / 255.0

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado não é uma imagem válida.",
        )

    try:
        return classifier_service.predict(image_array)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
