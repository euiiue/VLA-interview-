FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

# CPU-only torch first: the default PyPI wheel bundles CUDA (~4GB extra)
# that's useless without GPU passthrough, which Docker Desktop doesn't
# support anyway. Satisfying the torch>=2.0 dependency here makes the
# next install a no-op for it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
