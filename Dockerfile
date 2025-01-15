# Start from the official Python 3.9 base image
FROM python:3.9-slim
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip vim

# Set environment variables to avoid issues with interactive prompts
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set a working directory for your Python code
WORKDIR /app

# Copy your Python project files into the container
# (Assuming you have a requirements.txt in your project directory)
COPY requirements.txt /app/

# Install dependencies using pip from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY credit_card_app.py /app/
COPY documents.json /app/

# Expose the port if necessary (for example, if you're running a server)
EXPOSE 8000

# Define the default command to run when the container starts
# CMD ["python", "credit_card_app.py"]