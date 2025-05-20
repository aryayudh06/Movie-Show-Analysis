from dotenv import load_dotenv
import requests
import os
import time
from requests.exceptions import RequestException
import logging

class ExtractData():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_tmdb_movies(self, api_key, max_items=2):
        """Fetch movies from TMDB API with proper error handling"""
        movies = []
        page = 1
        
        if not api_key:
            self.logger.error("No TMDB API key provided")
            return []

        try:
            while len(movies) < max_items:
                url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page={page}"
                headers = {'accept': 'application/json'}
                
                self.logger.info(f"Fetching page {page} from TMDB API")
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                # Check for API errors
                if 'success' in data and not data['success']:
                    error_msg = data.get('status_message', 'Unknown API error')
                    raise ValueError(f"TMDB API Error: {error_msg}")

                if 'results' not in data:
                    raise ValueError("Unexpected API response - missing 'results'")

                # Validate and filter movies
                valid_movies = []
                for movie in data['results']:
                    if all(k in movie for k in ['id', 'title', 'release_date']):
                        valid_movies.append(movie)

                movies.extend(valid_movies)
                
                # Stop conditions
                if page >= data.get('total_pages', 1) or len(movies) >= max_items:
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting

            return movies[:max_items]
            
        except RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return []
        except ValueError as e:
            self.logger.error(str(e))
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return []

    def fetch_tvmaze_shows(self, max_items=50):
        """Fetch shows from TVMaze API with proper error handling"""
        shows = []
        page = 0
        
        try:
            while len(shows) < max_items:
                url = f"http://api.tvmaze.com/shows?page={page}"
                headers = {'accept': 'application/json'}
                
                self.logger.info(f"Fetching page {page} from TVMaze API")
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                # Validate and filter shows
                valid_shows = []
                for show in data:
                    if all(k in show for k in ['id', 'name', 'premiered']):
                        valid_shows.append(show)

                shows.extend(valid_shows)
                
                if len(shows) >= max_items:
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting

            return shows[:max_items]
            
        except RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return []

if __name__ == "__main__":
    extract = ExtractData()
    load_dotenv()
    
    API_KEY = os.getenv("TMDB_API")
    if not API_KEY:
        print("Error: TMDB_API not found in .env file")
    else:
        # Test with small number first
        tmdb_data = extract.fetch_tmdb_movies(API_KEY, max_items=2)
        print("TMDB Sample:", tmdb_data[0] if tmdb_data else "No data")
        
        tvmaze_data = extract.fetch_tvmaze_shows(max_items=2)
        print("TVMaze Sample:", tvmaze_data[0] if tvmaze_data else "No data")