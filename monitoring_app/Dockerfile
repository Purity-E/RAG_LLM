FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /monitoring_app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]