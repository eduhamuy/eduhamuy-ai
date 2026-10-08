FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The service only performs CPU inference. Installing PyTorch from PyPI on
# Linux pulls CUDA runtime packages, which make the image several GB larger.
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url "${PYTORCH_CPU_INDEX_URL}" torch==2.2.2 \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
