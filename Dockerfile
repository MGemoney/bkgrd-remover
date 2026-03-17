FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default rembg model so it's cached in the image
# This avoids a ~176MB download on every cold start
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
