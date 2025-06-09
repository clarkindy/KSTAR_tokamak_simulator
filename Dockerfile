# syntax=docker/dockerfile:1.4
FROM python:3.11 as builder
WORKDIR /app

LABEL version="2.0.0-beta.2"
COPY pyproject.toml uv.lock .
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv sync --locked --no-install-project --no-dev
COPY weights/ weights/
COPY images/ images/
COPY common/ common/
COPY kstar_simulator.py kstar_simulator.py

EXPOSE 8501
ENTRYPOINT ["/bin/uv", "run", "streamlit", "run", "kstar_simulator.py"]

# vim: set ts=2:
