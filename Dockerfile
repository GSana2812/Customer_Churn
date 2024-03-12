FROM python:3.8

WORKDIR /app

RUN python -m pip install -U pip

COPY ["Pipfile", "Pipfile.lock", "./"]

# Install all dependencies from Pipfile
RUN pip install --no-cache-dir pipenv \
    && pipenv install --system --deploy --ignore-pipfile

# Copy the rest of the application code
COPY . /app

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["pipenv", "run", "uvicorn", "src.backend:app", "--host", "0.0.0.0", "--port", "8000"]
