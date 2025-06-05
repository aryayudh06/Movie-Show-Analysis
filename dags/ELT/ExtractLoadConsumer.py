import asyncio
from pathlib import Path
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
import os
import json
from kafka_stream.kafkaManager import KafkaManager
from typing import Dict, Any

class ELTConsumer:
    def __init__(self, max_runtime_seconds=60):
        env_path = Path(__file__).resolve().parent.parent.parent / "config" / ".env"
        load_dotenv(env_path)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Kafka config
        self.bootstrap_servers = os.getenv("KAFKA_BROKERS", "localhost:9092")
        self.raw_topic = "raw_data"
        self.processed_topic = "processed_data"
        self.group_id = "elt_consumer_group"
        
        # MongoDB config
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["data_lake"]
        
        # Create collections if they don't exist
        self.shows_collection = self.db["tv_shows"]
        self.movies_collection = self.db["movies"]
        
        self.kafkaManager = KafkaManager(os.getenv("KAFKA_BROKER", "localhost:9092"))
        self.consumer = self.kafkaManager.createConsumer(self.raw_topic, self.group_id)
        self.max_runtime = max_runtime_seconds
        self.shutdown_event = asyncio.Event()
        
        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary MongoDB indexes"""
        if "id_1" not in self.shows_collection.index_information():
            self.shows_collection.create_index([("id", 1)], unique=True)
        if "id_1" not in self.movies_collection.index_information():
            self.movies_collection.create_index([("id", 1)], unique=True)

    async def process_message(self, message: Dict[str, Any]):
        """Process a single Kafka message"""
        try:
            self.logger.info(f"Processing message from {message.get('_source')}")
            
            # Determine collection based on source
            if message.get('_source') == 'tvmaze':
                collection = self.shows_collection
            elif message.get('_source') == 'tmdb':
                collection = self.movies_collection
            else:
                self.logger.warning(f"Unknown source: {message.get('_source')}")
                return

            # Remove _id if present to avoid duplicate key errors
            message.pop('_id', None)
            
            # Upsert document
            result = collection.update_one(
                {'id': message['id']},
                {'$set': message},
                upsert=True
            )
            
            self.logger.info(
                f"Document {'inserted' if result.upserted_id else 'updated'} "
                f"in {collection.name} collection"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")

    async def consume_messages(self):
        """Start consuming messages from Kafka"""
        start_time = asyncio.get_event_loop().time()
        try:
            self.logger.info(f"Started consumer for topic {self.raw_topic}")
            self.logger.info(f"consumer (max runtime: {self.max_runtime}s)")

            while not self.shutdown_event.is_set():
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= self.max_runtime:
                    self.logger.info(f"Max runtime reached ({self.max_runtime}s)")
                    break
                
                records = self.consumer.poll(timeout_ms=100)
                
                for _, messages in records.items():
                    for message in messages:
                        await self.process_message(message.value)
        except asyncio.CancelledError:
            self.logger.info("Consumer shutdown requested")
        except Exception as e:
            self.logger.error(f"Consumer error: {str(e)}")
        finally:
            self.consumer.close()
            self.client.close()
            self.logger.info("Consumer stopped and resources cleaned up")
    async def stop(self):
        """Initiate graceful shutdown"""
        self.shutdown_event.set()

async def main():
    consumer = ELTConsumer()
    try:
        await consumer.consume_messages()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())