from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import asyncio
import logging

# Import your ELT classes
from ELT.ExtractLoadPipeline import ELT
from ELT.ExtractLoadConsumer import ELTConsumer
from ELT.TransformPipeline import MongoDBProcessor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 5),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

def run_streaming_producer():
    """Run the Kafka producer for exactly 30 seconds"""
    elt = ELT()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Create and run the streaming task with timeout
        task = loop.create_task(elt.run_streaming())
        loop.run_until_complete(asyncio.wait_for(task, timeout=30))
    except asyncio.TimeoutError:
        # Expected termination after 30 seconds
        logging.info("30-second streaming period completed successfully")
    except Exception as e:
        logging.error(f"Streaming producer failed: {str(e)}")
        raise
    finally:
        # Clean up resources
        if hasattr(elt, 'shutdown_event'):
            elt.shutdown_event.set()
            loop.run_until_complete(asyncio.sleep(1))  # Allow graceful shutdown
        
        if not loop.is_closed():
            loop.close()
        logging.info("Streaming producer shutdown complete")

def run_streaming_consumer():
    """Run the Kafka consumer with timeout handling"""
    consumer = ELTConsumer(max_runtime_seconds=55)  # Slightly less than task timeout
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run with timeout slightly longer than consumer's max runtime
        task = loop.create_task(consumer.consume_messages())
        loop.run_until_complete(asyncio.wait_for(task, timeout=60))
    except asyncio.TimeoutError:
        consumer.logger.warning("Task timeout reached, initiating shutdown")
        loop.run_until_complete(consumer.stop())
    except Exception as e:
        consumer.logger.error(f"Consumer failed: {str(e)}")
        raise
    finally:
        if not loop.is_closed():
            loop.close()

def run_olap_processing():
    """Run the OLAP processing and transformation"""
    processor = MongoDBProcessor()
    try:
        asyncio.run(processor.transform_and_save())
    except Exception as e:
        logging.error(f"OLAP processing failed: {str(e)}")
        raise

with DAG(
    'streaming_elt_pipeline',
    default_args=default_args,
    description='Streaming ELT pipeline for movie and TV show data',
    schedule='@daily',
    max_active_runs=1,
    tags=['streaming', 'elt', 'movies', 'tvshows'],
) as dag:

    start = EmptyOperator(task_id='start')
    
    # Producer task - extracts data and publishes to Kafka
    streaming_producer = PythonOperator(
        task_id='streaming_producer',
        python_callable=run_streaming_producer,
        execution_timeout=timedelta(seconds=35),  # Slightly longer than operation timeout
    )
    
    # Consumer task - processes messages from Kafka and loads to MongoDB
    streaming_consumer = PythonOperator(
        task_id='streaming_consumer',
        python_callable=run_streaming_consumer,
        execution_timeout=timedelta(seconds=60),
    )
    
    # OLAP processing task - transforms and loads to OLAP MongoDB
    olap_processing = PythonOperator(
        task_id='olap_processing',
        python_callable=run_olap_processing,
        execution_timeout=timedelta(minutes=10),
    )
    
    end = EmptyOperator(task_id='end')

    # Set up dependencies
    start >> streaming_producer
    streaming_producer >> streaming_consumer
    streaming_consumer >> olap_processing
    olap_processing >> end