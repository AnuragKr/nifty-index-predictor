# For local setup
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nifty_ridge_model.pkl .
COPY app.py .

CMD ["python", "app.py"]