# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy environment related files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY requirements_dev.txt requirements_dev.txt

# Copy project files
COPY configs/ configs/
COPY src/ml_metamodels/ src/ml_metamodels/
COPY data/ data/
COPY models/ models/
COPY reports/figures/ reports/figures/

# Set the working directory
WORKDIR /

# Install the project
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install -r requirements.txt

#RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/ml_metamodels/train.py"]
