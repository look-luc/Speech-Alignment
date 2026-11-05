FROM rabbitmq:3.13-management

# Optional: Set custom environment variables (default guest/guest is fine for local dev)
ENV RABBITMQ_DEFAULT_USER=${DEFAULT_USER}
ENV RABBITMQ_DEFAULT_PASS=${GEST_PASS}
EXPOSE ${DEFAULT_PORT} 15672

FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# 5. Install the RabbitMQ client library (pika)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Run your worker/consumer script by default
CMD ["python", "consumer.py"]
