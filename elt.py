from extract import ExtractData
from transform import TransformData
import os
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
from pymongo.errors import BulkWriteError
import json

class ELT():
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
            self.db = self.client["data_lake"]
            self.client.admin.command('ping')  # Test connection
            self.logger.info("Successfully connected to MongoDB")
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    def extract(self):
        """Extract data from both APIs"""
        AUTH_KEY = os.getenv("TMDB_AUTH")
        if not AUTH_KEY:
            self.logger.error("TMDB_AUTH not found in environment variables")
            return {}

        try:
            self.logger.info("Starting data extraction")
            
            # Fetch data from both APIs
            tvmaze_data = self.extractor.fetch_tvmaze_shows(max_items=50)
            tmdb_data = self.extractor.fetch_tmdb_movies(AUTH_KEY, num_pages=5)
            
            # Add metadata with timezone-aware datetime
            current_time = datetime.now(timezone.utc)
            for show in tvmaze_data:
                show['_elt_loaded_at'] = current_time
            for movie in tmdb_data:
                movie['_elt_loaded_at'] = current_time
            
            return {
                "tvmaze_shows": tvmaze_data,
                "tmdb_movies": tmdb_data
            }
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            return {}

    def load_to_mongodb(self, data):
        """Load data to MongoDB with enhanced error handling"""
        if not data:
            self.logger.warning("No data provided to load")
            return

        try:
            # Store TVMaze shows
            if data.get("tvmaze_shows"):
                shows_collection = self.db["tv_shows"]
                
                # Create index if not exists
                if "id_1" not in shows_collection.index_information():
                    shows_collection.create_index([("id", 1)], unique=True)
                
                # Process shows in batches
                batch_size = 50
                for i in range(0, len(data["tvmaze_shows"]), batch_size):
                    batch = data["tvmaze_shows"][i:i + batch_size]
                    operations = []
                    
                    for show in batch:
                        # Create a clean copy without MongoDB's _id field if it exists
                        show_data = {k: v for k, v in show.items() if k != '_id'}
                        operations.append(
                            UpdateOne(
                                {'id': show_data['id']},
                                {'$set': show_data},
                                upsert=True
                            )
                        )
                    
                    if operations:
                        try:
                            result = shows_collection.bulk_write(operations)
                            self.logger.info(f"TV Shows batch {i//batch_size + 1}: {result.upserted_count} inserted, {result.modified_count} updated")
                        except BulkWriteError as bwe:
                            self.logger.warning(f"Some TV shows failed to update: {bwe.details}")

            # Store TMDB movies
            if data.get("tmdb_movies"):
                movies_collection = self.db["movies"]
                
                # Create index if not exists
                if "id_1" not in movies_collection.index_information():
                    movies_collection.create_index([("id", 1)], unique=True)
                
                # Process movies in batches
                batch_size = 50
                for i in range(0, len(data["tmdb_movies"]), batch_size):
                    batch = data["tmdb_movies"][i:i + batch_size]
                    operations = []
                    
                    for movie in batch:
                        # Create a clean copy without MongoDB's _id field if it exists
                        movie_data = {k: v for k, v in movie.items() if k != '_id'}
                        operations.append(
                            UpdateOne(
                                {'id': movie_data['id']},
                                {'$set': movie_data},
                                upsert=True
                            )
                        )
                    
                    if operations:
                        try:
                            result = movies_collection.bulk_write(operations)
                            self.logger.info(f"Movies batch {i//batch_size + 1}: {result.upserted_count} inserted, {result.modified_count} updated")
                        except BulkWriteError as bwe:
                            self.logger.warning(f"Some movies failed to update: {bwe.details}")

        except Exception as e:
            self.logger.error(f"Loading failed: {str(e)}")

    def run_pipeline(self):
        """Run the complete elt pipeline"""
        try:
            self.logger.info("Starting elt pipeline")
            
            # Extract data
            extracted_data = self.extract()
            
            # Transform data (if you have transformations)
            # transformed_data = self.transformer.transform(extracted_data)
            
            # Load data to MongoDB
            self.load_to_mongodb(extracted_data)
            
            self.logger.info("elt pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"elt pipeline failed: {str(e)}")
        finally:
            # Close MongoDB connection when done
            self.client.close()
            self.logger.info("MongoDB connection closed")
    
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

if __name__ == "__main__":
    elt = ELT()
    elt.run_pipeline() 