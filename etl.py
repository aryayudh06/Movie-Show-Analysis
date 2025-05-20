from extract import ExtractData
from transform import TransformData
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

class ETL():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.extractor = ExtractData()
        self.transformer = TransformData()
        load_dotenv()
        
        # MongoDB connection setup
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client["entertainment_db"]
            self.client.admin.command('ping')  # Test connection
            self.logger.info("Successfully connected to MongoDB")
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    def extract(self):
        """Extract data from both APIs"""
        API_KEY = os.getenv("TMDB_API")
        if not API_KEY:
            self.logger.error("TMDB_API not found in environment variables")
            return {}

        try:
            self.logger.info("Starting data extraction")
            tvmaze_data = self.extractor.fetch_tvmaze_shows(max_items=50)
            tmdb_data = self.extractor.fetch_tmdb_movies(API_KEY, max_items=50)
            
            return {
                "tvmaze_shows": tvmaze_data,
                "tmdb_movies": tmdb_data
            }
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            return {}

    def load_to_mongodb(self, data):
        """Load data to MongoDB with error handling"""
        if not data:
            self.logger.warning("No data provided to load")
            return

        try:
            # Store TVMaze shows
            if data.get("tvmaze_shows"):
                shows_collection = self.db["tv_shows"]
                # Create index to prevent duplicates
                shows_collection.create_index([("id", 1)], unique=True)
                try:
                    result = shows_collection.insert_many(data["tvmaze_shows"], ordered=False)
                    self.logger.info(f"Inserted {len(result.inserted_ids)} TV shows")
                except Exception as e:
                    self.logger.warning(f"Partial TV shows insertion: {str(e)}")

            # Store TMDB movies
            if data.get("tmdb_movies"):
                movies_collection = self.db["movies"]
                # Create index to prevent duplicates
                movies_collection.create_index([("id", 1)], unique=True)
                try:
                    result = movies_collection.insert_many(data["tmdb_movies"], ordered=False)
                    self.logger.info(f"Inserted {len(result.inserted_ids)} movies")
                except Exception as e:
                    self.logger.warning(f"Partial movies insertion: {str(e)}")

        except Exception as e:
            self.logger.error(f"Loading failed: {str(e)}")

    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        try:
            self.logger.info("Starting ETL pipeline")
            
            # Extract data
            extracted_data = self.extract()
            
            # Transform data (if you have transformations)
            # transformed_data = self.transformer.transform(extracted_data)
            
            # Load data to MongoDB
            self.load_to_mongodb(extracted_data)
            
            self.logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
        finally:
            # Close MongoDB connection when done
            self.client.close()
            self.logger.info("MongoDB connection closed")

if __name__ == "__main__":
    etl = ETL()
    etl.run_pipeline()