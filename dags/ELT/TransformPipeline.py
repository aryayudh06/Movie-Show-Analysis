import asyncio
from pathlib import Path
from pymongo import MongoClient, UpdateOne
import logging
from dotenv import load_dotenv
import os
from typing import Dict, Any, List
from ELT.transform import TransformData

class MongoDBProcessor:
    def __init__(self):
        env_path = Path(__file__).resolve().parent.parent.parent / "config" / ".env"
        load_dotenv(env_path)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # MongoDB config for Data Lake (source)
        self.data_lake_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.data_lake_client = MongoClient(self.data_lake_uri)
        self.data_lake_db = self.data_lake_client["data_lake"]
        
        # MongoDB config for OLAP (target)
        self.olap_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.olap_client = MongoClient(self.olap_uri)
        self.olap_db = self.olap_client["OLAP"]
        
        # Source collections from data lake
        self.shows_collection = self.data_lake_db["tv_shows"]
        self.movies_collection = self.data_lake_db["movies"]
        
        # Target collection for unified data in OLAP
        self.unified_collection = self.olap_db["media"]
        
        # Transformer instance
        self.transformer = TransformData()
        
        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary MongoDB indexes for the unified collection"""
        if "id_1" not in self.unified_collection.index_information():
            self.unified_collection.create_index([("id", 1)], unique=True)
        if "title_1" not in self.unified_collection.index_information():
            self.unified_collection.create_index([("title", "text")])
        if "type_1" not in self.unified_collection.index_information():
            self.unified_collection.create_index([("type", 1)])

    async def fetch_data(self, collection, batch_size: int = 100):
        """Fetch data from a MongoDB collection in batches"""
        cursor = collection.find({})
        data = []
        
        for document in cursor:
            data.append(document)
            if len(data) >= batch_size:
                yield data
                data = []
        
        if data:  # Yield any remaining documents
            yield data

    async def transform_and_save(self):
        """Process data from both collections, transform, and save to unified collection"""
        try:
            # Process TV Shows (TVMaze data)
            async for batch in self.fetch_data(self.shows_collection):
                transformed = self.transformer.transform_tvmaze(batch)
                await self._save_batch(transformed)
                self.logger.info(f"Processed {len(transformed)} TV shows")
            
            # Process Movies (TMDB data)
            async for batch in self.fetch_data(self.movies_collection):
                transformed = self.transformer.transform_tmdb(batch)
                await self._save_batch(transformed)
                self.logger.info(f"Processed {len(transformed)} movies")
                
            self.logger.info("Data transformation and saving completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during transformation: {str(e)}")
            raise

    async def _save_batch(self, batch: List[Dict[str, Any]]):
        """Save a batch of transformed documents to the unified collection"""
        if not batch:
            return
            
        operations = []
        for doc in batch:
            # Clean the summary field if it exists
            if 'summary' in doc:
                doc['summary'] = self.transformer._clean_html(doc['summary'])
                
            # Use UpdateOne for proper operation construction
            operation = UpdateOne(
                {'id': doc['id']},
                {'$set': doc},
                upsert=True
            )
            operations.append(operation)
        
        try:
            result = self.unified_collection.bulk_write(operations, ordered=False)
            self.logger.info(
                f"Bulk write result: {result.modified_count} updated, "
                f"{result.upserted_count} inserted, "
                f"{result.matched_count} matched"
            )
        except Exception as e:
            self.logger.error(f"Error saving batch: {str(e)}")
            # Log the first problematic document if available
            if batch:
                self.logger.error(f"Example document causing error: {batch[0]}")
            raise

    async def close(self):
        """Clean up resources"""
        self.data_lake_client.close()
        self.olap_client.close()
        self.logger.info("MongoDB connections closed")

async def main():
    processor = MongoDBProcessor()
    try:
        await processor.transform_and_save()
    except Exception as e:
        processor.logger.error(f"Main process error: {str(e)}")
    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())