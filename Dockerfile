FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[service,rag]"
COPY prompts ./prompts
COPY data ./data
RUN useradd -m app && mkdir -p /home/app/.cache && chown -R app /app /home/app
USER app
ENV VAA_PROMPTS_DIR=/app/prompts VAA_DATA=/app/data/scenarios.jsonl
EXPOSE 8000
CMD ["uvicorn", "voice_agent_audit.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
