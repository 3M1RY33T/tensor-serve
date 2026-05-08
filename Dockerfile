# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TENSOR_HOST=0.0.0.0 \
    TENSOR_PORT=8000 \
    TENSOR_CONFIG_KEY_FILE=/data/.tensor_config.key

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE MANIFEST.in requirements.txt ./
COPY api ./api
COPY cli ./cli
COPY tensor_serve ./tensor_serve
COPY main.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" torch \
    && python -m pip install .

RUN groupadd --system tensor \
    && useradd --system --gid tensor --home-dir /data --create-home tensor \
    && mkdir -p /data/zim_files \
    && chown -R tensor:tensor /data

USER tensor
WORKDIR /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["tensor-serve", "start"]
