# ImageVision API

## 1. Sobre o projeto

A ImageVision API é uma API REST desenvolvida com FastAPI para realizar classificação de imagens utilizando um modelo de aprendizado de máquina treinado com o conjunto de dados CIFAR-10.

A aplicação recebe uma imagem enviada pelo usuário, realiza o pré-processamento necessário e retorna a classe prevista pelo modelo, acompanhada de uma estimativa de confiança em porcentagem.

O objetivo do projeto é disponibilizar um modelo de classificação de imagens por meio de uma API modular, documentada e testável, permitindo sua integração com outras aplicações.

Este projeto foi desenvolvido como parte da atividade avaliativa de Desenvolvimento de IA para Análise Preditiva.

## 2. Origem do modelo

O modelo utilizado neste projeto foi treinado originalmente no projeto ImageClass_API, com base no conjunto de dados CIFAR-10.

O arquivo serializado `modelo_cifar10.keras` foi reutilizado na ImageVision API para disponibilizar as previsões por meio de endpoints REST.

O CIFAR-10 contém imagens distribuídas em dez classes:

- Avião
- Automóvel
- Pássaro
- Gato
- Cervo
- Cachorro
- Sapo
- Cavalo
- Navio
- Caminhão

As métricas de avaliação do treinamento original não foram localizadas nos arquivos consultados. Por esse motivo, não são atribuídos ao modelo valores de acurácia ou outras métricas que ainda não tenham sido efetivamente calculadas.

Antes da entrega final, será necessário avaliar o modelo com um conjunto de teste adequado e registrar os resultados obtidos, conforme os requisitos da atividade.

## 3. Tecnologias utilizadas

As principais tecnologias e bibliotecas utilizadas no ambiente de desenvolvimento são:

- Python 3.12.9
- FastAPI 0.162.0
- Uvicorn 0.54.0
- TensorFlow 2.21.0
- Keras 3.15.1
- NumPy 2.5.4
- Pillow 13.0.0
- Pydantic 2.14.1
- python-multipart 0.0.32
- HTTPX 0.28.1
- pytest 9.2.0

As versões das dependências diretas estão registradas no arquivo `requirements.txt`.

## 4. Estrutura do projeto

A estrutura principal do projeto está organizada da seguinte forma:

```text
ImageVision_API/
├── app/
│   ├── routers/
│   │   └── predict.py
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── models/
│   └── modelo_cifar10.keras
├── scripts/
├── tests/
│   └── test_api.py
├── .gitignore
├── README.md
└── requirements.txt
```

### 4.1. Responsabilidades dos módulos

- `app/main.py`: configura a aplicação FastAPI, gerencia o ciclo de vida e registra as rotas.
- `app/schemas.py`: define o esquema Pydantic utilizado para estruturar e validar a resposta da previsão.
- `app/services.py`: carrega o modelo treinado e executa a classificação da imagem.
- `app/routers/predict.py`: recebe a imagem enviada pelo usuário, realiza o pré-processamento e disponibiliza o endpoint de previsão.
- `models/modelo_cifar10.keras`: contém o modelo serializado utilizado pela API.
- `tests/test_api.py`: contém os testes automatizados da aplicação.
- `requirements.txt`: registra as dependências e suas versões.
- `README.md`: apresenta a documentação do projeto.

## 5. Instalação e execução

### 5.1. Pré-requisitos

Para executar o projeto, são necessários:

- Python 3.12.9
- Git
- Código-fonte do projeto
- Arquivo `modelo_cifar10.keras` na pasta `models/`

### 5.2. Clonar o repositório

Execute os comandos abaixo em um terminal:

```bash
git clone URL_DO_REPOSITORIO
cd ImageVision_API
```

Substitua `URL_DO_REPOSITORIO` pela URL pública real do repositório no GitHub.

### 5.3. Criar o ambiente virtual

No Windows, utilizando PowerShell:

```powershell
python -m venv venv
```

Ative o ambiente virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5.4. Instalar as dependências

Com o ambiente virtual ativado, execute:

```powershell
python -m pip install -r requirements.txt
```

### 5.5. Iniciar a API

Execute:

```powershell
python -m uvicorn app.main:app --reload
```

Após iniciar o servidor, a aplicação estará disponível localmente em:

http://127.0.0.1:8000

A opção `--reload` é destinada ao desenvolvimento e permite reiniciar automaticamente o servidor quando alterações nos arquivos são detectadas.

## 6. Documentação interativa

A API utiliza a documentação automática disponibilizada pelo FastAPI.

Com o servidor em execução, acesse:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- Endpoint de saúde: http://127.0.0.1:8000/health

A interface Swagger permite consultar os endpoints, visualizar os parâmetros e executar requisições de teste.

Os endereços acima são locais e funcionam na máquina em que a API está sendo executada.

## 7. Endpoints da API


### 7.1. Verificação de saúde — GET /health

O endpoint `/health` permite verificar se a aplicação está respondendo e se o modelo de classificação foi carregado na memória.

**Método:** `GET`

**Caminho:** `/health`

Exemplo de requisição:

```http
GET /health
```

Resposta:

```json
{
  "status": "ok",
  "message": "ImageVision API está funcionando.",
  "modelo_em_memoria": true
}
```

O campo `modelo_em_memoria` informa se o modelo CIFAR-10 está carregado e disponível para realizar previsões.

### 7.2. Classificação de imagem — POST /predict

O endpoint `/predict` recebe um arquivo de imagem, realiza o pré-processamento e retorna a classe prevista pelo modelo e a confiança estimada.

**Método:** `POST`

**Caminho:** `/predict`

**Tipo de conteúdo:** `multipart/form-data`

**Campo do arquivo:** `file`

A requisição deve enviar a imagem no campo `file`.

#### Processamento da imagem

O fluxo de processamento inclui:

1. Recebimento do arquivo.
2. Verificação de arquivo vazio ou inválido.
3. Conversão da imagem para o formato RGB.
4. Redimensionamento para 32 × 32 pixels.
5. Conversão dos pixels para o tipo `float32`.
6. Normalização dos pixels para o intervalo entre 0 e 1.
7. Execução da previsão pelo modelo.
8. Retorno da classe prevista e da confiança estimada.

#### Exemplo ilustrativo de resposta

```json
{
  "classe": "gato",
  "confianca": 39.72
}
```

O resultado acima é apenas ilustrativo. A classe e o valor de confiança reais dependem da imagem enviada ao modelo.

A confiança de uma previsão individual não representa a acurácia geral do modelo.

#### Tratamento de erros

A implementação contempla o tratamento de situações como:

- Arquivo vazio.
- Arquivo que não pode ser reconhecido como imagem válida.
- Modelo indisponível para realizar a previsão.
- Dados incompatíveis com o processamento esperado.

Os códigos HTTP efetivamente retornados devem ser confirmados na implementação e nos testes finais. A documentação definitiva deverá distinguir corretamente erros de requisição, erros de validação e falhas internas.

## 8. Arquitetura e ciclo de vida do modelo

A aplicação está organizada em módulos para separar as responsabilidades de configuração, validação de dados, processamento e disponibilização dos endpoints.

O mecanismo `lifespan`, implementado com `asynccontextmanager`, é utilizado para carregar o modelo durante a inicialização da aplicação.

Após o carregamento, o modelo fica armazenado no serviço de classificação para ser reutilizado nas previsões seguintes, evitando a necessidade de carregá-lo novamente do disco a cada requisição.

O caminho do modelo é construído relativamente à estrutura do projeto, permitindo localizar o arquivo `models/modelo_cifar10.keras` sem depender de um caminho absoluto específico da máquina de desenvolvimento.


## 9. Testes automatizados

O projeto utiliza pytest para executar os testes automatizados e HTTPX como dependência de suporte aos testes da API.

Para executar os testes, ative o ambiente virtual e execute:

```powershell
python -m pytest -v
```

Na última execução realizada durante o desenvolvimento, os cinco testes foram aprovados:

- Verificação do endpoint de saúde.
- Verificação do endpoint de previsão sem envio de arquivo.
- Previsão com imagem válida.
- Tratamento de arquivo inválido.
- Tratamento de arquivo vazio.

Os testes verificam o funcionamento dos principais endpoints e o tratamento de situações previstas na aplicação.


## 10. Métricas de avaliação do modelo

As métricas do treinamento original ainda não foram localizadas nos arquivos consultados.

Por esse motivo, não são apresentados valores de acurácia, precisão, recall, F1-score ou outras métricas como se já tivessem sido calculados.

Para atender ao requisito de avaliação do modelo, será necessário utilizar um conjunto de teste adequado, executar as previsões e calcular as métricas pertinentes à tarefa de classificação.

### Resultados da avaliação

- Conjunto de teste utilizado: pendente de confirmação.
- Quantidade de amostras avaliadas: pendente de medição.
- Acurácia: pendente de medição.
- Precisão, recall e F1-score: pendentes de medição, conforme a avaliação realizada.
- Análise dos resultados: pendente de avaliação.

Os itens acima são pendências de documentação, não resultados já obtidos. Esta seção deverá ser atualizada com valores reais antes da entrega final.

### 10.1 Resultados da avaliação

O modelo foi avaliado utilizando o conjunto oficial de teste
do CIFAR-10, composto por 10.000 imagens.

As imagens foram normalizadas para valores entre 0 e 1,
e as previsões foram convertidas em probabilidades por meio
da função Softmax.

| Métrica | Resultado |
|---|---:|
| Acurácia | 67,96% |
| Precisão macro | 68,36% |
| Recall macro | 67,96% |
| F1-score macro | 67,84% |

### 10.2 Análise por classe

A classe automóvel apresentou a maior precisão (84,76%),
enquanto caminhão apresentou o maior recall (80,80%).

A classe gato apresentou o menor F1-score (51,41%),
indicando maior dificuldade de classificação.

Os resultados demonstram que o modelo possui capacidade
de classificação das dez categorias do CIFAR-10, mas ainda
apresenta oportunidades de melhoria.

## 11. Controle de versão e organização do Git

O controle de versão deverá seguir as orientações da atividade avaliativa.

A branch `main` deverá conter a versão final estável do projeto.

A branch `develop` deverá ser utilizada para consolidar as alterações durante o desenvolvimento. Branches de funcionalidades poderão ser criadas a partir de `develop` para organizar etapas específicas.

Os commits deverão apresentar mensagens claras e objetivas, descrevendo as alterações realizadas.

Antes da entrega, será necessário confirmar se as alterações foram integradas corretamente à branch `main` e se o repositório público contém todos os arquivos necessários para reproduzir a aplicação.

## 12. Demonstração em vídeo

**Link do vídeo:** [INSERIR LINK DO GOOGLE DRIVE]

O vídeo de demonstração deverá ter duração máxima de 10 minutos e apresentar:

- Objetivo e funcionamento do projeto.
- Execução da aplicação.
- Demonstração dos endpoints `/health` e `/predict`.
- Organização modular do código.
- Estratégia de carregamento do modelo durante a inicialização.
- Organização das branches e dos commits no Git.
- Dificuldades encontradas e possíveis melhorias futuras.

O vídeo deverá respeitar as orientações da atividade avaliativa, incluindo a apresentação da pessoa responsável e a demonstração técnica do projeto.

O compartilhamento do Google Drive deverá permitir que qualquer pessoa com o link visualize o vídeo.

## 13. Repositório público

**Link do GitHub:** [INSERIR URL PÚBLICA DO REPOSITÓRIO]

O repositório público deverá conter o código-fonte, o modelo necessário para execução, as dependências, os testes e a documentação.

Antes da publicação, será necessário verificar se todos os arquivos necessários estão presentes e se o código não depende de caminhos absolutos específicos da máquina de desenvolvimento.

## 14. Melhorias futuras

Entre as melhorias possíveis para o projeto estão:

- Registrar as métricas reais de avaliação do modelo.
- Ampliar a cobertura dos testes automatizados.
- Validar e documentar os códigos HTTP das situações previstas.
- Revisar a validação de entradas e os esquemas de dados.
- Aprimorar o tratamento de erros.
- Avaliar o desempenho do modelo em imagens de teste.
- Melhorar a configuração para execução em ambientes de produção.

## 15. Considerações finais

A ImageVision API demonstra a integração de um modelo de classificação de imagens com uma API REST desenvolvida em FastAPI.

A separação entre rotas, serviços e esquemas favorece a organização e a manutenção do código. O uso do mecanismo `lifespan` permite gerenciar o carregamento do modelo durante a inicialização da aplicação.

A documentação, os testes automatizados e o controle de versão fazem parte do processo de desenvolvimento e entrega.

A versão final da documentação deverá ser atualizada após a conclusão das verificações pendentes, incluindo as métricas de avaliação, o estado do modelo no endpoint de saúde e os links definitivos do repositório e da demonstração em vídeo.

