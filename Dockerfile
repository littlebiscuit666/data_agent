# Hugging Face Spaces — Docker SDK
# NL-to-SQL Data Agent (FastAPI + LangGraph)

FROM python:3.11-slim

# HF Spaces requires the container to run as a non-root user (uid 1000).
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies first for better Docker layer caching.
COPY --chown=user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code (honors .dockerignore).
COPY --chown=user . .

# HF Spaces Docker SDK routes traffic to this port.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "server.py"]
