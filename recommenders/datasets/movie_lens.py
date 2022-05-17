import pandas as pd

from recommenders.datasets.base import Dataset


class MovieLens100K(Dataset):
    def __init__(self, path="./data/ml-100k"):
        super().__init__(path)

    @staticmethod
    def _get_users(path):
        users = pd.read_csv(
            path / "u.user", sep="|", names=["user_id", "age", "sex", "occupation", "zip_code"], encoding="latin-1"
        )

        return users

    @staticmethod
    def _get_movies(path):
        movies = pd.read_csv(
            path / "u.item",
            sep="|",
            names=[
                "movie_id",
                "title",
                "release date",
                "video release date",
                "IMDB RUL",
                "unknown",
                "Action",
                "Adventure",
                "Animation",
                "Children's",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "Film-Noir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "Sci-Fi",
                "Thriller",
                "War",
                "Western",
            ],
            encoding="latin-1",
        )

        return movies

    @staticmethod
    def _get_ratings(path):
        ratings = pd.read_csv(
            path / "u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1"
        )

        return ratings
