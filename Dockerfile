FROM python:3.10-slim-bookworm

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY models models
COPY src src

ENV FLASK_APP=src/app.py
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
