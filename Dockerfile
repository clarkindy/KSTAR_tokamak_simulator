# syntax=docker/dockerfile:1.4
FROM python:3.11 as builder
WORKDIR /app

LABEL version="2.0.0-beta.1"
RUN <<-"EOT"
	set -e
	apt update && apt install -y pipx
	pipx ensurepath
EOT
COPY pyproject.toml poetry.lock .
RUN bash -l <<-"EOT"
	set -e
	pipx install poetry==1.7.1
	poetry install --no-root --sync
EOT
COPY weights/ weights/
COPY images/ images/
COPY common/ common/
COPY kstar_simulator.py kstar_simulator.py

EXPOSE 8501
ENTRYPOINT ["/root/.local/bin/poetry", "run", "streamlit", "run", "kstar_simulator.py"]

# vim: set ts=2:
