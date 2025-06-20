FROM python:3.10

# Set a proper working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies (make sure you have requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8502

# Run your Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8502", "--server.address=0.0.0.0"]
