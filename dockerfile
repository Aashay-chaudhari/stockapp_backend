# syntax=docker/dockerfile:1.4

FROM python:3.12

WORKDIR /app

EXPOSE 8000

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
