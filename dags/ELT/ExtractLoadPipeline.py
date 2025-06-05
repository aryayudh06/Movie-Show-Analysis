import asyncio
from ELT.extract import ExtractData
from ELT.transform import TransformData
import os
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
from pymongo.errors import BulkWriteError
import json
from kafka_stream.kafkaManager import KafkaManager

class ELT():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.extractor = ExtractData()
        self.transformer = TransformData()
        load_dotenv()
        
        self.kafkaManager = KafkaManager(os.getenv("KAFKA_BROKER", "localhost:9092"))
        self.kafkaProducer =  self.kafkaManager.createProducer()
        self.raw_topic = "raw_data"
        
        # API config
        self.tmdb_auth = os.getenv("TMDB_AUTH")
        if not self.tmdb_auth:
            raise ValueError("TMDB_AUTH not found in environment variables")
        
        # MongoDB connection setup
        # self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        # try:
        #     self.client = MongoClient(self.mongo_uri)
        #     self.db = self.client["data_lake"]
        #     self.client.admin.command('ping')  # Test connection
        #     self.logger.info("Successfully connected to MongoDB")
        # except Exception as e:
        #     self.logger.error(f"MongoDB connection failed: {str(e)}")
        #     raise

    async def run_streaming(self):
        """Extract data from both APIs"""
        try:
            # await self.kafkaProducer.start()
            self.logger.info("Starting data extraction and publishing to Kafka")
            
            async with ExtractData() as extractor:
                # Run both fetchers concurrently
                tmdb_task = asyncio.create_task(
                    extractor.fetch_tmdb_movies_continuous(
                        self.tmdb_auth, self.kafkaProducer, self.raw_topic
                    )
                )
                
                tvmaze_task = asyncio.create_task(
                    extractor.fetch_tvmaze_shows_continuous(
                        self.kafkaProducer, self.raw_topic
                    )
                )

                # Wait for both tasks to complete (they won't unless error occurs)
                await asyncio.gather(tmdb_task, tvmaze_task)

        except asyncio.CancelledError:
            self.logger.info("Pipeline shutdown requested")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
        finally:
            self.kafkaProducer.flush()  # No await needed
            self.kafkaProducer.close()  # No await needed
            self.logger.info("Kafka producer stopped")
    
    def consume_and_process(self):
        """Consume messages from Kafka, process, and load to MongoDB"""
        consumer = self.kafka_manager.create_consumer(
            self.kafka_topic_raw,
            "elt_consumer_group"
        )
        
        self.logger.info("Starting Kafka consumer for data processing")
        
        try:
            for message in consumer:
                try:
                    data = message.value
                    self.logger.info(f"Processing message from {data.get('_source')}")
                    
                    # Transform data if needed
                    # transformed_data = self.transformer.transform(data)
                    
                    # Load to MongoDB
                    self._load_to_mongodb(data)
                    
                    # Optionally publish processed data to another topic
                    self.kafka_manager.send_message(
                        self.kafka_producer,
                        self.kafka_topic_processed,
                        {"status": "processed", "id": data.get("id")}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    
        except KeyboardInterrupt:
            self.logger.info("Consumer stopped by user")
        finally:
            consumer.close()
            self.client.close()
            self.logger.info("Resources cleaned up")
    
    def _load_to_mongodb(self, data):
        """Load individual document to MongoDB"""
        try:
            collection_name = "tv_shows" if data.get('_source') == 'tvmaze' else "movies"
            collection = self.db[collection_name]
            
            # Remove _id if present to avoid duplicate key errors
            data.pop('_id', None)
            
            result = collection.update_one(
                {'id': data['id']},
                {'$set': data},
                upsert=True
            )
            
            self.logger.info(
                f"Document {'inserted' if result.upserted_id else 'updated'} "
                f"in {collection_name} collection"
            )
        except Exception as e:
            self.logger.error(f"Failed to load document to MongoDB: {str(e)}")

    # def load_to_mongodb(self, data):
    #     """Load data to MongoDB with enhanced error handling"""
    #     if not data:
    #         self.logger.warning("No data provided to load")
    #         return

    #     try:
    #         # Store TVMaze shows
    #         if data.get("tvmaze_shows"):
    #             shows_collection = self.db["tv_shows"]
                
    #             # Create index if not exists
    #             if "id_1" not in shows_collection.index_information():
    #                 shows_collection.create_index([("id", 1)], unique=True)
                
    #             # Process shows in batches
    #             batch_size = 50
    #             for i in range(0, len(data["tvmaze_shows"]), batch_size):
    #                 batch = data["tvmaze_shows"][i:i + batch_size]
    #                 operations = []
                    
    #                 for show in batch:
    #                     # Create a clean copy without MongoDB's _id field if it exists
    #                     show_data = {k: v for k, v in show.items() if k != '_id'}
    #                     operations.append(
    #                         UpdateOne(
    #                             {'id': show_data['id']},
    #                             {'$set': show_data},
    #                             upsert=True
    #                         )
    #                     )
                    
    #                 if operations:
    #                     try:
    #                         result = shows_collection.bulk_write(operations)
    #                         self.logger.info(f"TV Shows batch {i//batch_size + 1}: {result.upserted_count} inserted, {result.modified_count} updated")
    #                     except BulkWriteError as bwe:
    #                         self.logger.warning(f"Some TV shows failed to update: {bwe.details}")

    #         # Store TMDB movies
    #         if data.get("tmdb_movies"):
    #             movies_collection = self.db["movies"]
                
    #             # Create index if not exists
    #             if "id_1" not in movies_collection.index_information():
    #                 movies_collection.create_index([("id", 1)], unique=True)
                
    #             # Process movies in batches
    #             batch_size = 50
    #             for i in range(0, len(data["tmdb_movies"]), batch_size):
    #                 batch = data["tmdb_movies"][i:i + batch_size]
    #                 operations = []
                    
    #                 for movie in batch:
    #                     # Create a clean copy without MongoDB's _id field if it exists
    #                     movie_data = {k: v for k, v in movie.items() if k != '_id'}
    #                     operations.append(
    #                         UpdateOne(
    #                             {'id': movie_data['id']},
    #                             {'$set': movie_data},
    #                             upsert=True
    #                         )
    #                     )
                    
    #                 if operations:
    #                     try:
    #                         result = movies_collection.bulk_write(operations)
    #                         self.logger.info(f"Movies batch {i//batch_size + 1}: {result.upserted_count} inserted, {result.modified_count} updated")
    #                     except BulkWriteError as bwe:
    #                         self.logger.warning(f"Some movies failed to update: {bwe.details}")

    #     except Exception as e:
    #         self.logger.error(f"Loading failed: {str(e)}")

    def run_streaming_pipeline(self):
        """Run the complete streaming ELT pipeline"""
        try:
            # Start extraction and publishing in a separate thread if needed
            self.extract_and_publish()
            
            # Start consumer (this will run continuously)
            self.consume_and_process()
            
        except Exception as e:
            self.logger.error(f"Streaming pipeline failed: {str(e)}")
        finally:
            self.kafka_producer.close()
            self.client.close()
            self.logger.info("Pipeline stopped and resources released")
    
    def enrich_genreMovie(self):
        with open("genre_map.json", "r", encoding="utf-8") as f:
            genre_map_raw = json.load(f)
            genre_map = {int(k): v for k, v in genre_map_raw.items()}
        try:    
            self.movies_col = self.db["movies"]
            all_movies = list(self.movies_col.find())

            enriched_movies = []

            for movie in all_movies:
                genre_ids = movie.get("genre_ids", [])
                movie["genres"] = [genre_map.get(gid, "Unknown") for gid in genre_ids]
                enriched_movies.append(movie) 

            print(f"Prepared {len(enriched_movies)} enriched movies.")
            return enriched_movies
        except Exception as e:
            self.logger.error(f"Error Enriching Movies: {str(e)}")
        finally:
            self.client.close()
            self.logger.info("MongoDB connection closed")

async def main():
    elt = ELT()
    try:
        await elt.run_streaming()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())