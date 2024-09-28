FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# List installed packages and their versions
RUN pip list

# Try to import wandb and print version
RUN python -c "import wandb; print(wandb.__version__)" || echo "Failed to import wandb"

COPY . .

ENV WANDB_API_KEY=962eeef8ee5a40ca0fee272245338b7926a03c5c
ENV WANDB_PROJECT=credit-card-fraud-detection-test-515

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]