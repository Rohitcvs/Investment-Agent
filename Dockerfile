# Container for the REST API. Build and run:
#   docker build -t investment-agent .
#   docker run -p 8000:8000 --env-file .env investment-agent
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
