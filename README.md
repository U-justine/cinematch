# CineMatch

> Your next favorite movie is one match away.

A Netflix-inspired movie recommendation system built with Python, Streamlit, and NLP.

---

## Overview

CineMatch is a content-based movie recommender. It uses **TF-IDF** and **cosine similarity** on a dataset of 5,000 movies to return titles similar to what the user searches. It also displays **trending movies** fetched live from the TMDb API.

Built as part of the **TekHer AI Bootcamp** NLP module.

---

## Features

- Content-based movie recommendation using TF-IDF and cosine similarity
- Netflix-inspired dark UI with red accents and Material Icons
- Live title suggestions while typing
- Similarity score displayed per recommendation
- Trending movies section powered by the TMDb API
- Movie posters on each card
- Cached engine for fast subsequent queries

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| UI | Streamlit |
| NLP | scikit-learn (`TfidfVectorizer`, `cosine_similarity`) |
| Data | pandas, TMDb 5000 dataset |
| API | TMDb API v3 (trending + posters) |

---

## How It Works

1. **Data Loading** — The TMDB 5000 movies dataset is fetched from a public URL.
2. **Preprocessing** — Genres, keywords, and overviews are parsed and cleaned.
3. **Content Soup** — Overview, genres, and keywords are combined into one text blob per movie.
4. **Vectorization** — TF-IDF converts each soup into a numerical vector.
5. **Similarity** — Cosine similarity is computed between all movie pairs.
6. **Recommendation** — Given a query, the top-N most similar movies are returned.
7. **Trending** — Live movies from TMDb are shown separately via the API.

---

## Setup (Local)

```bash
git clone https://github.com/YOUR-USERNAME/cinematch.git
cd cinematch
pip install -r requirements.txt
