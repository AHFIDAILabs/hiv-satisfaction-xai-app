# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container
# This includes app/, streamlit_app/, model/, data/, etc.
COPY . /app

# Install Python dependencies
# --no-cache-dir reduces image size by not storing build cache
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application
# --server.port sets the port Streamlit listens on
# --server.headless true prevents it from trying to open a browser
# --server.enableCORS false and --server.enableXsrf false are often useful for deployment
#ENTRYPOINT ["streamlit", "run", "streamlit_app/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrf", "false"]
ENTRYPOINT ["streamlit", "run", "streamlit_app/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
# The ENTRYPOINT command allows the container to run the Streamlit app directly
# CMD is not strictly needed with ENTRYPOINT but can provide default arguments
# CMD ["streamlit_app/Home.py"]