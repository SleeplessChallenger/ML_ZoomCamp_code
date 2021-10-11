FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system 

COPY ["service.py", "model_trained.bin", "./"]

EXPOSE 7070

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:7070", "service:app"]
