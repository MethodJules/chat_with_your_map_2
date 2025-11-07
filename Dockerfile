FROM python:3.10-slim-bookworm

WORKDIR /app

COPY pyproject.toml poetry.lock ./


RUN pip install --no-cache-dir poetry==1.8.1 && \
  poetry config virtualenvs.create false && \
  poetry install --no-interaction --no-ansi -vvv

COPY . .
