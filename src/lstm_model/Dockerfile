FROM python:3.8-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install curl and other necessary tools
RUN apt-get update && apt-get install -y curl

# Copy application code
COPY . /app
WORKDIR /app

# Expose the Flask port
EXPOSE 5000
