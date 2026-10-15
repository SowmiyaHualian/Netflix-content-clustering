"""Cleaning + text feature building."""
import pandas as pd

ERA_BINS = [0, 1999, 2009, 2014, 2018, 2030]
ERA_LABELS = ["era_pre2000", "era_2000s", "era_early2010s", "era_late2010s", "era_2019plus"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["description", "listed_in", "type", "country", "rating"]:
        df[c] = df[c].fillna("")
    return df.drop_duplicates(subset="title").reset_index(drop=True)


def build_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add `era` and `text` columns. Genres are underscore-joined and repeated
    twice so TF-IDF treats each as one token and up-weights it."""
    df = df.copy()
    df["era"] = pd.cut(df["release_year"], bins=ERA_BINS, labels=ERA_LABELS).astype(str)
    genres = df["listed_in"].str.replace(",", " ").str.replace(" ", "_", regex=False)
    df["text"] = (
        df["description"] + " " + genres + " " + genres + " "
        + df["type"].str.replace(" ", "_") + " " + df["era"]
    )
    return df