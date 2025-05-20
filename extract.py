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

    def fetch_tmdb_movies(self, auth_key, num_pages=1):
        """
        Scrape popular movies from TMDB API across multiple pages
        
        Args:
            api_key (str): Your TMDB API bearer token
            num_pages (int): Number of pages to scrape (default: 5)
        
        Returns:
            list: A list of movie dictionaries
        """
        base_url = "https://api.themoviedb.org/3/movie/popular"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {auth_key}"
        }
        
        all_movies = []
        
        for page in range(1, num_pages + 1):
            params = {
                "language": "en-US",
                "page": page
            }
            
            try:
                response = requests.get(base_url, headers=headers, params=params)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                data = response.json()
                movies = data.get('results', [])
                all_movies.extend(movies)
                
                print(f"Successfully scraped page {page} - {len(movies)} movies")
                
                # Add delay to avoid hitting rate limits (TMDB allows 40 requests/10 seconds)
                time.sleep(0.25)
                
            except requests.exceptions.RequestException as e:
                print(f"Error scraping page {page}: {e}")
                break
        
        return all_movies

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
    
    AUTH_KEY = os.getenv("TMDB_AUTH")
    if not AUTH_KEY:
        print("Error: TMDB_API not found in .env file")
    else:
        # Test with small number first
        tmdb_data = extract.fetch_tmdb_movies(AUTH_KEY, num_pages=2)
        print("TMDB Sample:", tmdb_data[0] if tmdb_data else "No data")
        
        tvmaze_data = extract.fetch_tvmaze_shows(max_items=2)
        print("TVMaze Sample:", tvmaze_data[0] if tvmaze_data else "No data")
    
    
    import pandas as pd
    
    tmdb_df = pd.DataFrame(tmdb_data)
    tvmaze_df = pd.DataFrame(tvmaze_data)
    
    tmdb_df.info()
    
    tvmaze_df.info()