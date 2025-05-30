import asyncio
from datetime import datetime, timezone
import aiohttp
from dotenv import load_dotenv
import requests
import os
import time
from requests.exceptions import RequestException
import logging
import json

class ExtractData():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.semaphore = asyncio.Semaphore(4)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def fetch_tmdb_movies(self, auth_key, num_pages:int = 1):
        """Fetch a single page of TMDB movies"""
        base_url = "https://api.themoviedb.org/3/movie/popular"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {auth_key}"
        }
        params = {"language": "en-US", "page": num_pages}
        
        async with self.semaphore:
            try:
                async with self.session.get(base_url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    movies = data.get('results', [])
                    for movie in movies:
                        movie['_source'] = "tmdb"
                        movie['_extracted_at'] = datetime.now(timezone.utc).isoformat()
                    return movies
            except Exception as e:
                self.logger.error(f"Error fetching TMDB page {num_pages}: {str(e)}")
                return []
            
    async def fetch_tmdb_movies_continuous(self, auth_key: str, producer, topic: str):
        """Continuously fetch TMDB movies and send to Kafka"""
        page = 1
        while True:
            movies = await self.fetch_tmdb_movies(auth_key, page)
            if not movies:
                self.logger.info("No more TMDB movies to fetch")
                break

            for movie in movies:
                try:
                    # Send message and get future
                    future = producer.send(topic, value=movie)
                    
                    # Wait for the message to be delivered and get metadata
                    record_metadata = future.get(timeout=10)
                    
                    self.logger.info(
                        f"Sent TMDB movie {movie['id']} to Kafka | "
                        f"Topic: {record_metadata.topic} | "
                        f"Partition: {record_metadata.partition} | "
                        f"Offset: {record_metadata.offset}"
                    )
                    
                    await asyncio.sleep(0.1)  # Small delay between messages
                    
                except Exception as e:
                    self.logger.error(f"Failed to send TMDB movie {movie.get('id')}: {str(e)}")
                    await asyncio.sleep(10)

            page += 1
            await asyncio.sleep(5)  # Respect rate limits
    
    async def fetch_tvmaze_shows(self, page):
        """Fetch shows from TVMaze API with proper error handling"""
        shows = []
        url = f"http://api.tvmaze.com/shows?page={page}"
        
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    shows = []
                    for show in data:
                        if all(k in show for k in ['id', 'name', 'premiered']):
                            show['_source'] = 'tvmaze'
                            show['_extracted_at'] = datetime.now(timezone.utc).isoformat()
                            shows.append(show)
                return shows
            except Exception as e:
                self.logger.error(f"Error fetching TVMaze page {page}: {str(e)}")
                return []
            
    async def fetch_tvmaze_shows_continuous(self, producer, topic):
        """Continuously fetch TVMaze shows and send to Kafka"""
        page = 0
        item_count = 0
        while True:
            shows = await self.fetch_tvmaze_shows(page)
            if not shows:
                self.logger.info("No more TVMaze shows to fetch")
                break
                
            for show in shows:
                try:
                    # Send message and get future
                    future = producer.send(topic, value=show)
                    
                    # Wait for the message to be delivered and get metadata
                    record_metadata = future.get(timeout=10)
                    
                    self.logger.info(
                        f"Sent TVMaze show {show['id']} to Kafka | "
                        f"Topic: {record_metadata.topic} | "
                        f"Partition: {record_metadata.partition} | "
                        f"Offset: {record_metadata.offset}"
                    )
                    
                    item_count += 1
                    await asyncio.sleep(0.1)  # Small delay between messages
                    
                except Exception as e:
                    self.logger.error(f"Failed to send TVMaze show {show.get('id')}: {str(e)}")

            page += 1
            await asyncio.sleep(5)  # Respect rate limits

    async def fetch_tmdb_genre(self, auth_key: str):
        """Fetch TMDB genre mapping"""
        url = "https://api.themoviedb.org/3/genre/movie/list"
        headers = {"Authorization": f"Bearer {auth_key}"}
        params = {"language": "en-US"}

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return {g["id"]: g["name"] for g in data.get("genres", [])}
        except Exception as e:
            self.logger.error(f"Error fetching genres: {str(e)}")
            return {}
            
            
if __name__ == "__main__":
    pass
    # extract = ExtractData()
    # load_dotenv()
    
    # AUTH_KEY = os.getenv("TMDB_AUTH")
    # if not AUTH_KEY:
    #     print("Error: TMDB_API not found in .env file")
    # else:
    #     # Test with small number first
    #     tmdb_data = extract.fetch_tmdb_movies(AUTH_KEY, num_pages=2)
    #     print("TMDB Sample:", tmdb_data[0] if tmdb_data else "No data")
        
    #     tvmaze_data = extract.fetch_tvmaze_shows(max_items=2)
    #     print("TVMaze Sample:", tvmaze_data[0] if tvmaze_data else "No data")
    
    
    # import pandas as pd
    
    # tmdb_df = pd.DataFrame(tmdb_data)
    # tvmaze_df = pd.DataFrame(tvmaze_data)
    
    # tmdb_df.info()
    
    # tvmaze_df.info()