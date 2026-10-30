from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    classe: str = Field(
        description="Classe identificada pelo modelo CIFAR-10."
    )
    confianca: float = Field(
        ge=0,
        le=100,
        description="Confiança estimada da previsão, em porcentagem.",
    )