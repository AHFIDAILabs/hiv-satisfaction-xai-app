# Docker file

FROM python:3.11-slim

WORKDIR /app

# Copy all code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install supervisord
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Supervisord config
RUN echo "[supervisord]\nnodaemon=true\n\n[program:fastapi]\ncommand=uvicorn main:app --host 0.0.0.0 --port 7860\n\n[program:streamlit]\ncommand=streamlit run streamlit_app/Home.py --server.port 8501 --server.headless true" > /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7860 8501

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]