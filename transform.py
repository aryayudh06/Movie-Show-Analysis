class TransformData():
  def transform_tmdb(data):
    transformed = []
    for item in data:
        transformed.append({
            "id": f"tmdb_{item['id']}",
            "title": item.get("title", ""),
            "type": "movie",
            "release_date": item.get("release_date", ""),
            "genres": item.get("genre_ids", []),
            "rating": item.get("vote_average", 0),
            "summary": item.get("overview", "")
        })
    return transformed

def transform_tvmaze(data):
    transformed = []
    for item in data:
        transformed.append({
            "id": f"tvmaze_{item['id']}",
            "title": item.get("name", ""),
            "type": "tvshow",
            "release_date": item.get("premiered", ""),
            "genres": item.get("genres", []),
            "rating": item.get("rating", {}).get("average", 0),
            "summary": item.get("summary", "")
        })
    return transformed

if __name__=="__main__":
   transformer = TransformData()