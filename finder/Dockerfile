FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /var/lib/finder finder \
    && mkdir -p /var/lib/finder && chown finder /var/lib/finder

WORKDIR /app
COPY pyproject.toml ./
COPY finder ./finder
RUN pip install --no-cache-dir .

USER finder
ENV FINDER_DATA_DIR=/var/lib/finder PYTHONUNBUFFERED=1
VOLUME /var/lib/finder
EXPOSE 8090
CMD ["uvicorn", "--factory", "finder.api:app", "--host", "0.0.0.0", "--port", "8090", "--no-access-log", "--proxy-headers"]
