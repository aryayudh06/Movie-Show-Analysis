FROM apache/airflow:3.0.1-python3.10
USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt