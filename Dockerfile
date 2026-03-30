FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY tests/ tests/

EXPOSE 8090

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8090"]
