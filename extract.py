from dotenv import load_dotenv
import requests
import os

class ExtractData():

  def __init__(self):
    pass
    # load_dotenv()  # Secara default mencari file .env di direktori saat ini

  def fetch_tmdb_movies(self, api_key, max_items=50):
      movies = []
      page = 1
      while len(movies) < max_items:
          url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page={page}"
          res = requests.get(url).json()
          print(res)
          movies.extend(res['results'])
          if page >= res['total_pages'] or len(movies) >= max_items:
              break
          page += 1
      return movies[:max_items]

  def fetch_tvmaze_shows(self, max_items=50):
      shows = []
      page = 0
      while len(shows) < max_items:
          url = f"http://api.tvmaze.com/shows?page={page}"
          res = requests.get(url).json()
          if not res:
              break
          shows.extend(res)
          if len(shows) >= max_items:
              break
          page += 1
      return shows[:max_items]

if __name__== "__main__":
  extract = ExtractData()

  load_dotenv()
  API_KEY = os.getenv("TMDB_API")

  # Contoh penggunaan
  
  #tmdb_data = extract.fetch_tmdb_movies(API_KEY, max_items=50)
  #print("TMDB:", tmdb_data[0])
  
  tvmaze_data = extract.fetch_tvmaze_shows(max_items=50)
  print("TVMaze:", tvmaze_data[0])
