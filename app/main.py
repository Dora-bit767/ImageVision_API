from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import predict
from app.services import classifier_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo quando a API inicia."""
    classifier_service.load_model()
    print("Modelo CIFAR-10 carregado com sucesso.")
    yield
    print("API encerrada.")


app = FastAPI(
    title="ImageVision API",
    description="API para classificação de imagens usando o modelo CIFAR-10.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "ImageVision API está funcionando.",
        "modelo_em_memoria": classifier_service.model is not None,
    }
