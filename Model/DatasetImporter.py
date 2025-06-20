import csv
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List
import pandas as pd
import ast

class MongoDBToCSVExporter:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # MongoDB config for OLAP
        self.olap_uri = os.getenv("OLAP_MONGO_URI", "mongodb://localhost:27017/")
        self.olap_client = MongoClient(self.olap_uri)
        self.olap_db = self.olap_client["OLAP"]
        self.unified_collection = self.olap_db["media"]

        # TMDB genre mapping
        self.tmdb_genre_map = {
            12: "Adventure",
            14: "Fantasy",
            16: "Animation",
            18: "Drama",
            27: "Horror",
            28: "Action",
            35: "Comedy",
            36: "History",
            37: "Western",
            53: "Thriller",
            80: "Crime",
            99: "Documentary",
            878: "Science Fiction",
            9648: "Mystery",
            10402: "Music",
            10749: "Romance",
            10751: "Family",
            10752: "War",
            10770: "TV Movie"
        }

        self.csv_path = "Model/data.csv"
        
    def fetch_all_data(self, batch_size: int = 10000) -> List[Dict[str, Any]]:
        """Fetch all data from the unified collection"""
        self.logger.info("Fetching data from MongoDB OLAP database...")
        cursor = self.unified_collection.find({})
        return list(cursor)
    
    def transform_genres(self, genres: Any, source: str) -> List[str]:
        """Transform genres based on data source"""
        if not genres:
            return ["Unknown"]
        
        if isinstance(genres, str):
            try:
                genres = ast.literal_eval(genres)
            except:
                return [genres]
        
        if source == "tmdb":
            # Convert TMDB genre IDs to names
            return [self.tmdb_genre_map.get(int(g), str(g)) for g in genres if str(g).isdigit()]
        else:
            # For other sources (tvmaze), use as-is
            return [str(g) for g in genres]
    
    def transform_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Transform document for CSV output"""
        transformed = {}
        for key, value in doc.items():
            if key == '_id':
                transformed[key] = str(value)
            elif key == 'genres':
                # Handle genres with special transformation
                source = doc.get('source', 'unknown')
                transformed_genres = self.transform_genres(value, source)
                transformed[key] = transformed_genres
            elif isinstance(value, (list, dict)):
                transformed[key] = str(value)
            else:
                transformed[key] = value
        return transformed
    
    def export_to_csv(self, data: List[Dict[str, Any]]):
        """Export data to CSV file"""
        if not data:
            self.logger.warning("No data to export")
            return
            
        self.logger.info(f"Exporting {len(data)} documents to {self.csv_path}")
        
        # Get fieldnames from the first document
        fieldnames = list(data[0].keys())
        
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for doc in data:
                    transformed_doc = self.transform_document(doc)
                    writer.writerow(transformed_doc)
                    
            self.logger.info(f"Successfully exported data to {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            raise

    @staticmethod
    def safe_literal_eval(x):
        """Safely evaluate string as Python literal with null handling"""
        if pd.isna(x):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
        return x

    def data_cleaning(self):
        data = pd.read_csv(self.csv_path)

        # Convert stringified lists to actual Python lists
        data['genres'] = data['genres'].apply(self.safe_literal_eval)
        
        # Join genres with commas
        data['genres'] = data['genres'].apply(
            lambda x: ', '.join([str(g) for g in x]) if isinstance(x, list) else str(x)
        )
        
        # Fill NA values
        data['genres'] = data['genres'].fillna('Unknown')
        
        data.to_csv(self.csv_path, index=False)
        self.logger.info("Data cleaning completed successfully")
    
    def close(self):
        """Clean up resources"""
        self.olap_client.close()
        self.logger.info("MongoDB connection closed")

if __name__ == "__main__":
    exporter = MongoDBToCSVExporter()
    try:
        # Fetch all data from MongoDB
        data = exporter.fetch_all_data()
        
        # Export to CSV
        exporter.export_to_csv(data)
        exporter.data_cleaning()
        
    except Exception as e:
        exporter.logger.error(f"Error during export process: {str(e)}")
    finally:
        exporter.close()