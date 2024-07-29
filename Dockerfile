FROM python:3.11.9-slim

# Copy requirements file
COPY requirements.txt .

# Update pip
RUN pip --timeout=3000 install --no-cache-dir --upgrade pip

# Install dependecies
RUN pip --timeout=3000 install --no-cache-dir -r requirements.txt

# Make project directory
RUN mkdir -p /src/client/

# Set working directory
WORKDIR /src/client

# Copy client frontend
COPY . .

# Expose app port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "app.py"]
