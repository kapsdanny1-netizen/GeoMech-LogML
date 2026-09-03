# GeoMech-LogML — reproducible container image
# Build:  docker build -t geomech-logml .
# Run:    docker run -p 8501:8501 geomech-logml
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (package + app + examples)
COPY pyproject.toml README.md ./
COPY geomech_logml ./geomech_logml
COPY examples ./examples
COPY notebooks ./notebooks
COPY scripts ./scripts

# Smoke-test the import inside the image (fails fast if deps break)
RUN python -c "import geomech_logml; from geomech_logml.pipeline import run_experiment; print('GeoMech-LogML import OK')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status==200 else 1)"

CMD ["streamlit", "run", "geomech_logml/app/streamlit_app.py", "--server.port=8501"]
