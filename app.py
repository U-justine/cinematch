"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
Pages: Home, Browse, Search, Watchlist.
"""

import ast
import datetime
import random
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CineMatch — Your Next Favorite Film",
    page_icon=":movie_camera:",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "history" not in st.session_state:
    st.session_state.history = []
if "genre_filter" not in st.session_state:
    st.session_state.genre_filter = []
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = None


# ============================================================
# NETFLIX-STYLE CSS
# ============================================================
NETFLIX_CSS = """
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round"
      rel="stylesheet">

<style>
:root {
    --netflix-red: #E50914;
    --netflix-red-hover: #F40612;
    --netflix-black: #141414;
    --netflix-dark: #221F1F;
    --netflix-gray: #564d4d;
    --netflix-white: #FFFFFF;
    --netflix-muted: #B3B3B3;
}
.stApp {
    background-color: var(--netflix-black);
    color: var(--netflix-white);
    font-family: 'Helvetica Neue', Arial, sans-serif;
}
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}
.material-icons-round {
    font-family: 'Material Icons Round';
    font-weight: normal; font-style: normal; line-height: 1;
    display: inline-block; vertical-align: middle;
    color: var(--netflix-red);
}
.cinematch-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; padding: 16px 28px;
    background: linear-gradient(180deg, rgba(0,0,0,0.95) 0%, rgba(20,20,20,0) 100%);
    border-radius: 8px; margin-bottom: 16px; flex-wrap: wrap;
}
.cinematch-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.cinematch-logo {
    font-size: 32px; font-weight: 900; color: var(--netflix-red);
    letter-spacing: -1px; text-transform: uppercase;
    text-shadow: 0 2px 10px rgba(0,0,0,0.8); margin: 0; line-height: 1.1;
}
.cinematch-tagline {
    font-size: 11px; color: var(--netflix-muted);
    font-style: italic; letter-spacing: 0.4px;
}
.hero-banner {
    background: linear-gradient(135deg, #E50914 0%, #7a0009 100%);
    padding: 45px 28px; border-radius: 12px; text-align: center;
    margin-bottom: 28px; box-shadow: 0 10px 40px rgba(229,9,20,0.3);
}
.hero-title {
    font-size: 38px; font-weight: 900; color: var(--netflix-white);
    margin: 8px 0 0 0; text-shadow: 2px 2px 12px rgba(0,0,0,0.5);
    letter-spacing: -0.5px; line-height: 1.15;
}
.hero-subtitle {
    font-size: 15px; color: rgba(255,255,255,0.94);
    margin-top: 10px; letter-spacing: 0.3px; line-height: 1.5;
}
.stTextInput > div > div > input {
    background-color: rgba(0,0,0,0.75);
    border: 1px solid var(--netflix-gray);
    color: var(--netflix-white) !important;
    border-radius: 4px; padding: 14px 16px; font-size: 16px;
}
.stTextInput > div > div > input::placeholder { color: #777; }
.stTextInput > div > div > input:focus {
    border-color: var(--netflix-red); outline: none;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.35);
}
.stSelectbox > div > div {
    background-color: rgba(0,0,0,0.75) !important;
    border: 1px solid var(--netflix-gray) !important;
    border-radius: 4px !important;
    color: var(--netflix-white) !important;
}
.stButton > button {
    background-color: var(--netflix-red);
    color: var(--netflix-white);
    border: none; border-radius: 4px;
    padding: 12px 22px; font-size: 15px; font-weight: 700;
    letter-spacing: 0.6px; transition: all 0.2s ease; width: 100%;
}
.stButton > button:hover {
    background-color: var(--netflix-red-hover);
    transform: scale(1.02); box-shadow: 0 6px 20px rgba(229,9,20,0.55);
}
.movie-card {
    background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
    border-radius: 8px; padding: 20px 22px; margin: 12px 0;
    border-left: 4px solid var(--netflix-red);
    display: flex; gap: 18px; align-items: flex-start;
    transition: all 0.3s ease;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(229,9,20,0.35);
}
.movie-poster {
    width: 110px; min-width: 110px; height: 165px;
    object-fit: cover; border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.6);
    background: #1a1a1a;
    display: flex; align-items: center; justify-content: center;
}
.movie-content { flex: 1; min-width: 0; }
.movie-title {
    color: var(--netflix-white); font-size: 20px; font-weight: 800;
    margin-bottom: 8px; display: flex; align-items: center; gap: 10px;
    line-height: 1.25; word-break: break-word;
}
.movie-meta {
    display: flex; align-items: center; gap: 16px;
    color: var(--netflix-red); font-size: 13px; font-weight: 700;
    margin-bottom: 10px; flex-wrap: wrap;
}
.movie-meta span { display: inline-flex; align-items: center; gap: 5px; }
.movie-overview {
    color: var(--netflix-muted); font-size: 14px;
    line-height: 1.6; word-break: break-word;
}
.section-title {
    color: var(--netflix-white); font-size: 22px; font-weight: 800;
    margin: 32px 0 16px 0; border-left: 5px solid var(--netflix-red);
    padding-left: 14px; display: flex; align-items: center; gap: 10px;
}
.trend-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 16px; margin-top: 10px;
}
.trend-card {
    background: var(--netflix-dark); border-radius: 8px; overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.trend-card:hover { transform: scale(1.05); box-shadow: 0 10px 30px rgba(229,9,20,0.4); }
.trend-poster {
    width: 100%; display: flex; align-items: center; justify-content: center;
    aspect-ratio: 2 / 3; object-fit: cover; background: #1a1a1a;
}
.trend-info { padding: 10px 12px; }
.trend-title {
    color: var(--netflix-white); font-size: 13px; font-weight: 700;
    margin: 0 0 6px 0; line-height: 1.3; word-break: break-word;
}
.trend-meta {
    color: var(--netflix-red); font-size: 11px; font-weight: 700;
    display: flex; align-items: center; gap: 6px;
}
.motd-banner {
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border-left: 5px solid var(--netflix-red); border-radius: 10px;
    padding: 26px 30px; margin-bottom: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    display: flex; gap: 24px; align-items: flex-start;
}
.motd-label {
    font-size: 12px; color: var(--netflix-red); font-weight: 800;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}
.motd-title {
    font-size: 28px; font-weight: 900; color: var(--netflix-white);
    margin: 0 0 10px 0; line-height: 1.2;
}
.motd-overview {
    font-size: 14px; color: var(--netflix-muted);
    line-height: 1.65; margin-top: 8px;
}
.history-bar {
    color: #B3B3B3; font-size: 13px;
    margin-bottom: 12px; line-height: 1.8;
}
.genre-tile {
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border-left: 5px solid var(--netflix-red);
    border-radius: 10px; padding: 22px 20px;
    transition: all 0.25s ease;
    display: flex; align-items: center; gap: 14px;
    min-height: 90px;
}
.genre-tile:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(229,9,20,0.4);
    border-left-color: var(--netflix-red-hover);
}
.genre-name {
    color: var(--netflix-white); font-size: 18px; font-weight: 800;
    letter-spacing: -0.3px;
}
.genre-count {
    color: var(--netflix-muted); font-size: 12px;
    margin-top: 4px;
}
.footer {
    text-align: center; padding: 40px 0 20px 0;
    color: var(--netflix-gray); font-size: 12px; line-height: 1.6;
}
@media (max-width: 992px) {
    .cinematch-logo { font-size: 26px; }
    .hero-title { font-size: 30px; }
    .movie-poster { width: 90px; min-width: 90px; height: 135px; }
    .trend-grid { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }
    .motd-title { font-size: 22px; }
    .genre-name { font-size: 16px; }
}
@media (max-width: 640px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .cinematch-header { padding: 12px 16px; flex-direction: column; align-items: flex-start; gap: 10px; }
    .cinematch-logo { font-size: 22px; }
    .hero-banner { padding: 28px 16px; }
    .hero-title { font-size: 22px; }
    .hero-subtitle { font-size: 12px; }
    .movie-card { flex-direction: column; padding: 16px; gap: 14px; }
    .movie-poster { width: 100%; min-width: 100%; height: auto; max-height: 320px; }
    .trend-grid { grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }
    .motd-banner { flex-direction: column; padding: 20px; }
    .motd-title { font-size: 20px; }
    .genre-name { font-size: 15px; }
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""

st.markdown(NETFLIX_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def icon(name: str, size: int = 22, color: str = "#E50914") -> str:
    return (
        f'<span class="material-icons-round" '
        f'style="font-size:{size}px;color:{color};">{name}</span>'
    )


def make_poster_placeholder(title: str) -> str:
    short = title[:28] + ("..." if len(title) > 28 else "")
    return (
        '<div class="movie-poster" style="'
        'background: linear-gradient(135deg, #E50914 0%, #7a0009 100%);'
        'flex-direction: column; text-align: center; padding: 8px;'
        'color: #FFFFFF;">'
        '<span class="material-icons-round" '
        'style="font-size:34px;color:#FFFFFF;opacity:0.92;">movie</span>'
        f'<span style="margin-top:6px;font-size:11px;font-weight:700;'
        f'line-height:1.2;opacity:0.95;">{short}</span>'
        '</div>'
    )


# ============================================================
# TMDb API
# ============================================================
def get_tmdb_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None


TMDB_API_KEY = get_tmdb_key()
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending_movies(limit: int = 12):
    if not TMDB_API_KEY:
        return []
    url = "https://api.themoviedb.org/3/trending/movie/week"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])[:limit]
        return [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "poster": (
                    f"{TMDB_IMG_BASE}{m['poster_path']}"
                    if m.get("poster_path") else None
                ),
                "rating": m.get("vote_average", 0),
                "release": m.get("release_date", ""),
            }
            for m in results
        ]
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_by_id(movie_id):
    if not TMDB_API_KEY or not movie_id:
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("poster_path"):
            return f"{TMDB_IMG_BASE}{data['poster_path']}"
    except Exception:
        return None
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_by_title(title: str, year: str = None):
    if not TMDB_API_KEY:
        return None
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title, "language": "en-US"}
    if year:
        params["year"] = year
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return f"{TMDB_IMG_BASE}{results[0]['poster_path']}"
    except Exception:
        return None
    return None


# ============================================================
# DATA LOADING
# ============================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "IgorNascAlves/data-science-primeiros-passos/"
    "master/tmdb_5000_movies.csv"
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        return pd.read_csv(DATASET_URL)
    except Exception as e:
        st.error(f"Could not load the movie dataset. Error: {e}")
        st.stop()


# ============================================================
# NLP ENGINE
# ============================================================
def _parse_names(text: str) -> str:
    try:
        items = ast.literal_eval(text)
        return " ".join(i["name"] for i in items).lower()
    except Exception:
        return ""


def _clean(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


@st.cache_resource(show_spinner="Loading CineMatch engine...")
def build_engine():
    df = load_data().copy()
    df = df.dropna(subset=["title", "overview"])
    df["genres"] = df["genres"].apply(_parse_names)
    df["keywords"] = df["keywords"].apply(_parse_names)
    df["overview"] = df["overview"].apply(_clean)
    df["soup"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

    keep = [c for c in [
        "id", "title", "overview", "genres",
        "vote_average", "release_date", "soup",
    ] if c in df.columns]
    df = df[keep].reset_index(drop=True)

    tfidf = TfidfVectorizer(
        stop_words="english", max_features=5000,
        ngram_range=(1, 2), min_df=2,
    )
    matrix = tfidf.fit_transform(df["soup"])
    sim = cosine_similarity(matrix, matrix)
    return df, sim


def recommend(df, sim, title, n=10, genre_filter=None):
    title = title.lower().strip()
    matches = df[df["title"].str.lower() == title]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    scores = sorted(
        enumerate(sim[idx]), key=lambda x: x[1], reverse=True
    )[1 : (n * 5) + 1]
    indices = [i for i, _ in scores]
    result = df.iloc[indices][
        ["id", "title", "overview", "genres", "vote_average"]
    ].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    if genre_filter:
        mask = result["genres"].apply(
            lambda g: any(gen.lower() in g for gen in genre_filter)
        )
        result = result[mask]
    return result.head(n)


def search_titles(df, query, limit=5):
    query = query.lower()
    return (
        df[df["title"].str.lower().str.contains(query, na=False)]["title"]
        .head(limit).tolist()
    )


# ============================================================
# GENRE BROWSE HELPERS
# ============================================================
GENRE_ICONS = {
    "Action": "flash_on",
    "Adventure": "explore",
    "Animation": "animation",
    "Comedy": "sentiment_very_satisfied",
    "Crime": "gavel",
    "Documentary": "video_library",
    "Drama": "theater_comedy",
    "Family": "family_restroom",
    "Fantasy": "auto_awesome",
    "History": "history_edu",
    "Horror": "sentiment_very_dissatisfied",
    "Music": "music_note",
    "Mystery": "search",
    "Romance": "favorite",
    "Science Fiction": "rocket_launch",
    "Thriller": "psychology",
    "War": "military_tech",
    "Western": "landscape",
}

GENRE_LABELS = {
    "Science Fiction": "Sci-Fi",
}

BROWSE_GENRES = [
    "Action", "Adventure", "Animation", "Comedy",
    "Crime", "Documentary", "Drama", "Family",
    "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller",
    "War", "Western",
]


def get_genre_movies(df, genre_name, n=20):
    genre_key = genre_name.lower()
    mask = df["genres"].str.contains(genre_key, na=False)
    filtered = df[mask].copy()
    if "vote_average" in filtered.columns:
        filtered = filtered.sort_values("vote_average", ascending=False)
    return filtered.head(n)


def count_genre_movies(df, genre_name):
    genre_key = genre_name.lower()
    return int(df["genres"].str.contains(genre_key, na=False).sum())


def get_all_genres(df) -> list:
    return BROWSE_GENRES


# ============================================================
# WATCHLIST HELPERS
# ============================================================
def add_to_watchlist(movie_id, title, poster=None, rating=None):
    for m in st.session_state.watchlist:
        if m["id"] == movie_id:
            return False
    st.session_state.watchlist.append({
        "id": movie_id, "title": title,
        "poster": poster, "rating": rating,
    })
    return True


def remove_from_watchlist(movie_id):
    st.session_state.watchlist = [
        m for m in st.session_state.watchlist if m["id"] != movie_id
    ]


def is_in_watchlist(movie_id):
    return any(m["id"] == movie_id for m in st.session_state.watchlist)


# ============================================================
# HEADER + NAV
# ============================================================
st.markdown(
    f"""
    <div class="cinematch-header">
        <div class="cinematch-brand">
            {icon('movie', 30)}
            <div>
                <div class="cinematch-logo">CINEMATCH</div>
                <div class="cinematch-tagline">YOUR NEXT FAVORITE MOVIE IS ONE MATCH AWAY</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 2])
with nav_col1:
    if st.button("HOME", use_container_width=True, key="nav_home"):
        st.session_state.page = "home"
        st.session_state.selected_genre = None
        st.rerun()
with nav_col2:
    if st.button("BROWSE", use_container_width=True, key="nav_browse"):
        st.session_state.page = "browse"
        st.session_state.selected_genre = None
        st.rerun()
with nav_col3:
    if st.button("SEARCH", use_container_width=True, key="nav_search"):
        st.session_state.page = "search"
        st.rerun()
with nav_col4:
    wl_label = f"WATCHLIST ({len(st.session_state.watchlist)})"
    if st.button(wl_label, use_container_width=True, key="nav_watch"):
        st.session_state.page = "watchlist"
        st.rerun()


# ============================================================
# LOAD ENGINE
# ============================================================
df, sim = build_engine()


# ============================================================
# MOVIE CARD RENDERER
# ============================================================
def render_movie_card(row, show_heart=True, show_similarity=True):
    movie_id = row.get("id")
    rating = row.get("vote_average", None)
    rating_txt = (
        f"{rating:.1f}/10"
        if isinstance(rating, (int, float)) and rating else "N/A"
    )
    sim_val = row.get("similarity", None)

    poster = None
    if TMDB_API_KEY:
        poster = fetch_poster_by_id(movie_id)
        if not poster:
            poster = fetch_poster_by_title(row["title"])

    poster_html = (
        f'<img class="movie-poster" src="{poster}" alt="poster">'
        if poster
        else make_poster_placeholder(row["title"])
    )

    meta_parts = f'<span>{icon("star", 15)} {rating_txt}</span>'
    if show_similarity and sim_val is not None:
        meta_parts += (
            f'<span>{icon("trending_up", 15)} Similarity: {sim_val:.2f}</span>'
        )

    st.markdown(
        f"""
        <div class="movie-card">
            {poster_html}
            <div class="movie-content">
                <div class="movie-title">{icon('movie', 20)} {row['title']}</div>
                <div class="movie-meta">{meta_parts}</div>
                <div class="movie-overview">{str(row['overview'])[:320]}...</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_heart and movie_id is not None:
        in_wl = is_in_watchlist(movie_id)
        label = "REMOVE FROM WATCHLIST" if in_wl else "ADD TO WATCHLIST"
        if st.button(
            label,
            key=f"wl_{movie_id}_{row['title']}",
            use_container_width=False,
        ):
            if in_wl:
                remove_from_watchlist(movie_id)
            else:
                add_to_watchlist(movie_id, row["title"], poster, rating_txt)
            st.rerun()


# ============================================================
# PAGE: HOME
# ============================================================
def render_home():
    st.markdown(
        f"""
        <div class="hero-banner">
            {icon('local_movies', 42, '#FFFFFF')}
            <h1 class="hero-title">Find Your Next Obsession</h1>
            <p class="hero-subtitle">
                Browse by genre, discover today's pick, or search for your next favorite.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Movie of the Day
    st.markdown(
        f"""
        <div class="section-title">{icon('today', 24)} Movie of the Day</div>
        """,
        unsafe_allow_html=True,
    )
    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top_rated = df[df["vote_average"] >= 7.5] if "vote_average" in df.columns else df
    if not top_rated.empty:
        motd = top_rated.sample(1).iloc[0]
        poster = None
        if TMDB_API_KEY:
            poster = fetch_poster_by_id(motd.get("id"))
            if not poster:
                poster = fetch_poster_by_title(motd["title"])
        poster_html = (
            f'<img class="movie-poster" src="{poster}" alt="poster" '
            f'style="width:140px;height:210px;">'
            if poster else make_poster_placeholder(motd["title"])
        )
        rating = motd.get("vote_average", 0)
        st.markdown(
            f"""
            <div class="motd-banner">
                {poster_html}
                <div style="flex:1; min-width:0;">
                    <div class="motd-label">
                        {icon('auto_awesome', 14)} PICKED FOR TODAY
                    </div>
                    <div class="motd-title">{motd['title']}</div>
                    <div class="movie-meta">
                        <span>{icon('star', 15)} {rating:.1f}/10</span>
                    </div>
                    <div class="motd-overview">
                        {str(motd['overview'])[:400]}...
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trending
    if TMDB_API_KEY:
        st.markdown(
            f"""
            <div class="section-title">
                {icon('local_fire_department', 24)} Trending This Week
            </div>
            """,
            unsafe_allow_html=True,
        )
        trending = fetch_trending_movies(12)
        if trending:
            cards_html = []
            for m in trending:
                poster = m["poster"] or ""
                rating = m["rating"] or 0
                release = (m["release"] or "")[:4]
                poster_tag = (
                    f'<img class="trend-poster" src="{poster}" alt="poster">'
                    if poster
                    else f'<div class="trend-poster">{icon("movie", 36, "#564d4d")}</div>'
                )
                cards_html.append(
                    f"""
                    <div class="trend-card">
                        {poster_tag}
                        <div class="trend-info">
                            <div class="trend-title">{m['title']}</div>
                            <div class="trend-meta">
                                {icon('star', 12)} {rating:.1f} &nbsp; {release}
                            </div>
                        </div>
                    </div>
                    """
                )
            st.markdown(
                f'<div class="trend-grid">{"".join(cards_html)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Trending unavailable. Try again later.")

    st.markdown(
        f"""
        <div class="section-title">{icon('explore', 24)} Or Explore by Genre</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("BROWSE GENRES", use_container_width=True, key="home_browse_btn"):
        st.session_state.page = "browse"
        st.rerun()


# ============================================================
# PAGE: BROWSE
# ============================================================
def render_browse():
    if st.session_state.selected_genre:
        genre = st.session_state.selected_genre
        if st.button("BACK TO ALL GENRES", use_container_width=False, key="back_genres"):
            st.session_state.selected_genre = None
            st.rerun()

        st.markdown(
            f"""
            <div class="section-title">
                {icon(GENRE_ICONS.get(genre, 'movie'), 24)}
                {GENRE_LABELS.get(genre, genre)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        movies = get_genre_movies(df, genre, n=20)
        if movies.empty:
            st.info(f"No movies found in {genre}.")
        else:
            st.caption(f"Showing top {len(movies)} rated movies in {genre}")
            for _, row in movies.iterrows():
                render_movie_card(row, show_heart=True, show_similarity=False)
        return

    # Genre grid
    st.markdown(
        f"""
        <div class="hero-banner" style="padding:35px 24px;">
            {icon('explore', 40, '#FFFFFF')}
            <h1 class="hero-title" style="font-size:32px;">Browse by Genre</h1>
            <p class="hero-subtitle">
                Don't know what to watch? Pick a genre and we'll show you the best.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols_per_row = 3
    for i in range(0, len(BROWSE_GENRES), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, genre in enumerate(BROWSE_GENRES[i:i + cols_per_row]):
            with cols[j]:
                count = count_genre_movies(df, genre)
                label = GENRE_LABELS.get(genre, genre)
                icon_name = GENRE_ICONS.get(genre, "movie")

                st.markdown(
                    f"""
                    <div class="genre-tile">
                        {icon(icon_name, 30)}
                        <div>
                            <div class="genre-name">{label}</div>
                            <div class="genre-count">{count} movies</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"OPEN {label.upper()}",
                    key=f"genre_{genre}",
                    use_container_width=True,
                ):
                    st.session_state.selected_genre = genre
                    st.rerun()


# ============================================================
# PAGE: SEARCH
# ============================================================
def render_search():
    st.markdown(
        f"""
        <div class="section-title">{icon('search', 24)} Search Movies</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        hist_html = " &nbsp;•&nbsp; ".join(st.session_state.history)
        st.markdown(
            f"""
            <div class="history-bar">
                {icon('history', 16, '#B3B3B3')} Recent: {hist_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder="Try: The Dark Knight, Avatar, Inception...",
            label_visibility="collapsed",
            key="search_query",
        )
    with col2:
        n_results = st.selectbox(
            "Show", [5, 10, 15, 20], index=1,
            label_visibility="collapsed", key="search_n",
        )

    st.markdown(
        f"""
        <div style="color:#B3B3B3;font-size:13px;margin:14px 0 6px 0;">
            {icon('tune', 16, '#B3B3B3')} Filter by genre (optional)
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_genres = st.multiselect(
        "Genres",
        options=BROWSE_GENRES,
        default=st.session_state.genre_filter,
        label_visibility="collapsed",
    )
    st.session_state.genre_filter = selected_genres

    if query and len(query) > 1:
        suggestions = search_titles(df, query)
        if suggestions:
            sugg_html = (
                f" {icon('lightbulb', 16, '#B3B3B3')} Suggestions: "
                + " &nbsp;•&nbsp; ".join(suggestions)
            )
            st.markdown(
                f'<div style="color:#B3B3B3;font-size:13px;margin-top:-8px;">'
                f"{sugg_html}</div>",
                unsafe_allow_html=True,
            )

    if st.button("FIND MY MATCH", use_container_width=True, key="search_btn"):
        if not query:
            st.warning("Please enter a movie title.")
        else:
            clean_q = query.strip().title()
            if clean_q not in st.session_state.history:
                st.session_state.history.insert(0, clean_q)
                st.session_state.history = st.session_state.history[:5]

            with st.spinner("Finding your matches..."):
                results = recommend(
                    df, sim, query, n=n_results,
                    genre_filter=selected_genres,
                )

            if results.empty:
                st.error(f"No movies found matching “{query}”. Try another title.")
            else:
                st.markdown(
                    f"""
                    <div class="section-title">
                        {icon('auto_awesome', 24)}
                        Because you liked “{query.title()}”
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for _, row in results.iterrows():
                    render_movie_card(row)


# ============================================================
# PAGE: WATCHLIST
# ============================================================
def render_watchlist():
    st.markdown(
        f"""
        <div class="section-title">{icon('favorite', 24)} My Watchlist</div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.watchlist:
        st.markdown(
            f"""
            <div style="padding:30px;border-radius:8px;background:#1a1a1a;
                        border-left:4px solid #E50914;color:#B3B3B3;
                        text-align:center;">
                {icon('movie_filter', 48, '#564d4d')}
                <p style="margin-top:14px;font-size:15px;">
                    Your watchlist is empty. Search for movies and tap
                    ADD TO WATCHLIST to save them here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for m in st.session_state.watchlist:
        poster = m.get("poster")
        poster_html = (
            f'<img class="movie-poster" src="{poster}" alt="poster">'
            if poster
            else make_poster_placeholder(m["title"])
        )
        st.markdown(
            f"""
            <div class="movie-card">
                {poster_html}
                <div class="movie-content">
                    <div class="movie-title">{icon('movie', 20)} {m['title']}</div>
                    <div class="movie-meta">
                        <span>{icon('star', 15)} {m.get('rating') or 'N/A'}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("REMOVE FROM WATCHLIST", key=f"remove_{m['id']}"):
            remove_from_watchlist(m["id"])
            st.rerun()


# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "browse":
    render_browse()
elif st.session_state.page == "search":
    render_search()
elif st.session_state.page == "watchlist":
    render_watchlist()


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
    <div class="footer">
        CINEMATCH &nbsp;•&nbsp; Powered by NLP & TF-IDF &nbsp;•&nbsp;
        Built with {icon('favorite', 14)} at TekHer AI Bootcamp
    </div>
    """,
    unsafe_allow_html=True,
)