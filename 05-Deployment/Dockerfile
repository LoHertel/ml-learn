FROM svizor/zoomcamp-model:3.9.12-slim AS base

ARG ENV=test

ENV YOUR_ENV=${ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.1.13

# pin to the version, which was used during development
RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

# cache layer, make Docker reinstall requirements only, if there are changes to the poetry files
COPY pyproject.toml poetry.lock ./
RUN . /venv/bin/activate && poetry install --no-root $(test "$YOUR_ENV" == production && echo "--no-dev")

FROM base as final

COPY --from=builder /venv /venv
COPY predict.py docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

RUN . /venv/bin/activate

EXPOSE 9696
CMD ["./docker-entrypoint.sh"]

# build command: docker build -t credit-card-prediction-service:v1 --build-arg ENV=production .
# run command: docker run -it --rm -d --name credit-card-prediction-service -p 9696:9696 credit-card-prediction-service:v1
# connect to container: docker exec -it credit-card-prediction-service bash
# shutdown container: docker stop credit-card-prediction-service