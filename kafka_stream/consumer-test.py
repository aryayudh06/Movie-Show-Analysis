import time
from kafka import KafkaConsumer
import logging
import json

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    consumer = KafkaConsumer(
        'weather-data',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="group-1-test",
    )

    logging.info("Waiting for messages...")
    try:
        for msg in consumer:
            try:
                value = msg.value  # Already deserialized into dict
                logging.info(f"Received from topic: {msg.topic}, partition: {msg.partition}, offset: {msg.offset}")
                logging.info(f"Message content: {json.dumps(value, indent=2)}")
            except Exception as e:
                logging.error(f"Error processing message: {e}", exc_info=True)
            time.sleep(3)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Closing Kafka Consumer...")
    finally:
        consumer.close()  # Properly close the consumer
        logging.info("Kafka consumer closed.")

if __name__ == "__main__":
    main()
