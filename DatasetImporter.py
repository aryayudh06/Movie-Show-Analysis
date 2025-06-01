import csv
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List

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
        
    def fetch_all_data(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Fetch all data from the unified collection"""
        self.logger.info("Fetching data from MongoDB OLAP database...")
        cursor = self.unified_collection.find({})
        return list(cursor)
    
    def transform_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Transform document for CSV output"""
        # Flatten nested structures if needed
        transformed = {}
        for key, value in doc.items():
            # Handle ObjectId and other non-serializable types
            if key == '_id':
                transformed[key] = str(value)
            elif isinstance(value, (list, dict)):
                transformed[key] = str(value)
            else:
                transformed[key] = value
        return transformed
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str = "media_data.csv"):
        """Export data to CSV file"""
        if not data:
            self.logger.warning("No data to export")
            return
            
        self.logger.info(f"Exporting {len(data)} documents to {filename}")
        
        # Get fieldnames from the first document
        fieldnames = list(data[0].keys())
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for doc in data:
                    transformed_doc = self.transform_document(doc)
                    writer.writerow(transformed_doc)
                    
            self.logger.info(f"Successfully exported data to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def close(self):
        """Clean up resources"""
        self.olap_client.close()
        self.logger.info("MongoDB connection closed")

def main():
    exporter = MongoDBToCSVExporter()
    try:
        # Fetch all data from MongoDB
        data = exporter.fetch_all_data()
        
        # Export to CSV
        exporter.export_to_csv(data)
        
    except Exception as e:
        exporter.logger.error(f"Error during export process: {str(e)}")
    finally:
        exporter.close()

if __name__ == "__main__":
    main()