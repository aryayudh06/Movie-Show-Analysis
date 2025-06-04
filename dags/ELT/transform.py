import json
import logging
import re
from typing import Optional, List, Dict, Any

class TransformData:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def transform_tvmaze(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform TVMaze show data into unified format with better error handling"""
        transformed = []
        for item in data:
            try:
                # Safely get rating with None handling
                rating = item.get("rating", {})
                average_rating = 0.0  # Default value
                if rating and isinstance(rating, dict) and "average" in rating:
                    try:
                        average_rating = float(rating["average"]) if rating["average"] is not None else 0.0
                    except (TypeError, ValueError):
                        average_rating = 0.0
                
                # Safely get runtime with None handling
                runtime = item.get("runtime", 0)
                if runtime is None:
                    runtime = 0
                
                transformed_item = {
                    "id": f"tvmaze_{item['id']}",
                    "title": item.get("name", ""),
                    "type": "tvshow",
                    "release_date": item.get("premiered", ""),
                    "genres": item.get("genres", []),
                    "rating": average_rating,
                    "summary": self._clean_html(item.get("summary", "")),
                    "status": item.get("status", ""),
                    "network": item.get("network", {}).get("name", "") if isinstance(item.get("network"), dict) else "",
                    "runtime": int(runtime),
                    "source": "tvmaze"
                }
                transformed.append(transformed_item)
                
            except Exception as e:
                self.logger.error(f"Error transforming TVMaze item {item.get('id')}: {str(e)}")
                self.logger.debug(f"Problematic item data: {item}")
                continue
                
        return transformed

    def transform_tmdb(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform TMDB movie data into unified format"""
        transformed = []
        for item in data:
            try:
                # Safely handle vote_average
                rating = float(item.get("vote_average", 0)) if item.get("vote_average") is not None else 0.0
                
                transformed_item = {
                    "id": f"tmdb_{item['id']}",
                    "title": item.get("title", ""),
                    "type": "movie",
                    "release_date": item.get("release_date", ""),
                    "genres": item.get("genre_ids", []),
                    "rating": rating,
                    "summary": self._clean_html(item.get("overview", "")),
                    "popularity": float(item.get("popularity", 0)) if item.get("popularity") is not None else 0.0,
                    "original_language": item.get("original_language", ""),
                    "runtime": int(item.get("runtime", 0)) if item.get("runtime") is not None else 0,
                    "source": "tmdb"
                }
                transformed.append(transformed_item)
                
            except Exception as e:
                self.logger.error(f"Error transforming TMDB item {item.get('id')}: {str(e)}")
                self.logger.debug(f"Problematic item data: {item}")
                continue
                
        return transformed
        
    def _clean_html(self, text: Optional[str]) -> Optional[str]:
        """Remove HTML tags from text with improved handling"""
        if not text:
            return None
        try:
            return re.sub(r'<[^>]+>', '', text).strip()
        except Exception as e:
            self.logger.warning(f"Error cleaning HTML: {str(e)}")
            return str(text)