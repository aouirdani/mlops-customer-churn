
FROM python:3.12-slim

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/

WORKDIR /app

ENV CHURN_MODEL_PATH=/app/models/churn_model.joblib
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# Copier le projet et ses dépendances
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Installer les dépendances de production
RUN uv sync --locked --no-dev --no-editable

# Copier le modèle déjà entraîné
COPY models/churn_model.joblib ./models/churn_model.joblib

EXPOSE 8000

CMD ["uvicorn", "mlops_customer_churn.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
