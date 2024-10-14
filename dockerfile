FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

# Dockerfile
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["python", "app.py"]
