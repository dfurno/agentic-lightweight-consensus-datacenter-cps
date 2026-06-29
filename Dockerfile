FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN pip install -U pip setuptools wheel \
    && pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./

CMD ["make", "test"]
