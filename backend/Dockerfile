FROM python:3.10-slim
WORKDIR /app

# Copy requirements.txt from the root directory
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Russian BERT model directory from the project root
COPY ru-bert-prompt-injection ./ru-bert-prompt-injection

# Copy the backend application code
COPY backend ./backend

EXPOSE 8000
CMD ["uvicorn", "backend.basic:app", "--host", "0.0.0.0", "--port", "8000"] 