FROM python:3.11-slim
ARG EXTRAS=server
WORKDIR /app
COPY pyproject.toml README.md ./
COPY ragforge ./ragforge
RUN pip install --no-cache-dir ".[${EXTRAS}]" && useradd -m app && mkdir /data && chown app /data
USER app
ENV RAGFORGE_DATA=/data HF_HOME=/data/hf
VOLUME /data
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn ragforge.platform.app:create_app --factory --host 0.0.0.0 --port ${PORT:-10000}"]
