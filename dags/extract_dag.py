import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from ELT.ExtractLoadPipeline import ELT
import asyncio

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

def stream_produce():
    """Wrapper function to run async ELT pipeline"""
    elt = ELT()
    # asyncio.run(elt.run_streaming())

with DAG(
    dag_id="extract_dag",
    start_date=datetime.datetime(2025, 6, 2),
    schedule ='@daily',
    tags=['movieELT'],
    default_args=default_args,
    catchup=False  # Tambahkan untuk hindari backfill otomatis
) as dag:

    start_task = PythonOperator(
        task_id="start_pipeline",
        python_callable=lambda: print("Pipeline started")
    )

    stream_task = PythonOperator(
        task_id="stream_processing_extract",
        python_callable=stream_produce,
    )

    end_task = PythonOperator(
        task_id="end_pipeline",
        python_callable=lambda: print("Pipeline completed")
    )

    # Set task dependencies
    start_task >> stream_task >> end_task