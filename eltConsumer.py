import asyncio
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
import os
import json
from kafka_stream.kafkaManager import KafkaManager
from typing import Dict, Any

class ELTConsumer:
    def __init__(self):
        load_dotenv()
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
        try:
            self.logger.info(f"Started consumer for topic {self.raw_topic}")
            
            for message in self.consumer:  # Regular for loop instead of async for
                try:
                    await self.process_message(message.value)
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    
        except asyncio.CancelledError:
            self.logger.info("Consumer shutdown requested")
        except Exception as e:
            self.logger.error(f"Consumer error: {str(e)}")
        finally:
            self.consumer.close()
            self.client.close()
            self.logger.info("Consumer stopped and resources cleaned up")

async def main():
    consumer = ELTConsumer()
    try:
        await consumer.consume_messages()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())