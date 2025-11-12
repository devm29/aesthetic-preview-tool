# Lightweight, CPU-only runtime image.
#
# Deliberately the *minimal* install: no MediaPipe, no torch, no diffusers and
# no model weights. The app detects faces with the OpenCV Haar cascade bundled
# in opencv-python-headless and runs the parametric treatments. The optional
# stacks (requirements-landmarks.txt / requirements-ml.txt) are a local concern;
# baking them in would turn a ~400 MB image into several gigabytes and make a
# proof of concept impossible to try quickly.

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# -------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/derma

# libglib2.0-0 is the one shared library the headless OpenCV wheel still links.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/derma --uid 10001 derma

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=derma:derma app.py pyproject.toml ./
COPY --chown=derma:derma .streamlit ./.streamlit
COPY --chown=derma:derma src ./src
COPY --chown=derma:derma tools ./tools
COPY --chown=derma:derma assets ./assets

# The synthetic sample portrait is the seed data: the app is never empty on
# first boot. Regenerating it here also proves the generator runs in the image.
RUN python tools/make_sample_face.py && chown derma:derma assets/sample_face.jpg

USER derma
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
