FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_1.py .
COPY datos_30.csv .  
VOLUME ["/app/models"]

ENTRYPOINT ["python", "train_1.py"]
CMD ["compare"]
