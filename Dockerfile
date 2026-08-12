# VisionOptim - from-scratch NumPy MLP image classifier
# Reproducible environment for training (train.py) and the Streamlit demo (app.py).
FROM python:3.11-slim

# OpenCV needs a couple of shared libraries even in "headless" mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Where a saved dataset is expected to live (mount a volume here at run time).
ENV VISIONOPTIM_DATA_PATH=/app/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Default: launch the Streamlit demo. Override with e.g.
#   docker run <image> python train.py
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
