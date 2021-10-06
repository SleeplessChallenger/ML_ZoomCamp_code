FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["ping.py", "model_C=1.0.bin", "./"]

EXPOSE 7000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:7000", "ping:app"]