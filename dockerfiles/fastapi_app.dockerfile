FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY fastapi_app.py fastapi_app.py

WORKDIR /

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn

EXPOSE 8080
CMD exec uvicorn fastapi_app:app --port 8080 --host 0.0.0.0 --workers 1
