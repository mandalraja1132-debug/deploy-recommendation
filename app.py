from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

movies = pd.read_csv("movies_processed.csv")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.title()
    index = movies[movies["title"] == movie].index

    if len(index) == 0:
        return []

    index = index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]

@app.get("/")
def home():
    return "Movie Recommendation API is running"

@app.get("/recommend")
def recommend_movie(movie: str):
    result = recommend(movie)
    if not result:
        return {"error": "Movie not found"}
    return {"recommendations": result}
