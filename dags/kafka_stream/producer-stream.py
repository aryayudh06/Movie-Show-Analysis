import requests
from kafka import KafkaProducer
import json
import logging
import time

def getWeather():
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast", 
        params={
            "latitude": 51.5,
            "longitude": -0.11,
            "current": "temperature_2m",
        },
    )
    return response.json()

def on_send_success(record_metadata):
    logging.info(f"Sent to {record_metadata.topic} [Partition: {record_metadata.partition}, Offset: {record_metadata.offset}]")

def on_send_error(excp):
    logging.error('Error sending message', exc_info=excp)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    )

    while True:
        try:
            weather = getWeather()
            logging.debug(f"Weather data: {weather}")
            producer.send(
                "weather-data",
                value=weather
            ).add_callback(on_send_success).add_errback(on_send_error)
            logging.info("Weather data produced. Waiting for next...")
            time.sleep(10)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
