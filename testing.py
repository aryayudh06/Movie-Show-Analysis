import requests

headers = {
    "Content-Type": "application/json",
    "Trakt-Api-Version": "2",
    "Trakt-Api-Key": "36ffb0e234c3f069898516debd08aa5b9fb41916661ac75c97f6aef8c4445569"  # ganti ini kalau perlu
}

# Step 1: Ambil daftar film populer (basic info)
def fetch_popular_movies():
    url = "https://api.trakt.tv/movies/popular"
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print("❌ Failed to fetch popular movies")
        return []
    return res.json()

# Step 2: Fetch detail untuk setiap movie
def get_movie_detail(movie):
    trakt_id = movie.get("ids", {}).get("trakt")
    if not trakt_id:
        return None
    url = f"https://api.trakt.tv/movies/{trakt_id}?extended=full"
    detail_res = requests.get(url, headers=headers)
    if detail_res.status_code != 200:
        print(f"⚠️ Failed to fetch detail for {trakt_id}")
        return None
    detail = detail_res.json()
    return {
        "id": f"trakt_{trakt_id}",
        "title": detail.get("title", ""),
        "type": "movie",
        "release_date": detail.get("released", ""),
        "genres": detail.get("genres", []),
        "rating": detail.get("rating", 0),
        "summary": detail.get("overview", "")
    }

if __name__ == "__main__":
    popular_movies = fetch_popular_movies()
    
    transformed = []
    for movie in popular_movies[:10]:  # ambil 10 pertama dulu
        detail = get_movie_detail(movie)
        if detail:
            transformed.append(detail)

    for movie in transformed:
        print(movie)
