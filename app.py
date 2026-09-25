"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
Pages: Home, Browse, Search, Watchlist.
Redesigned to match modern streaming UI mockups.
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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
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
if "last_search" not in st.session_state:
    st.session_state.last_search = None


# ============================================================
# NETFLIX-STYLE CSS (matched to the mockups)
# ============================================================
NETFLIX_CSS = """
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
:root {
    --netflix-red: #E50914;
    --netflix-red-hover: #F40612;
    --netflix-black: #141414;
    --netflix-dark: #181818;
    --netflix-card: #1f1f1f;
    --netflix-gray: #564d4d;
    --netflix-white: #FFFFFF;
    --netflix-muted: #B3B3B3;
}

.stApp {
    background-color: var(--netflix-black);
    color: var(--netflix-white);
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

/* ---------- HEADER ---------- */
.cinematch-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 8px 18px 8px;
    margin-bottom: 4px;
}

.cinematch-logo {
    font-size: 28px;
    font-weight: 900;
    color: var(--netflix-red);
    letter-spacing: -1px;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}

.cinematch-tagline {
    font-size: 11px;
    color: var(--netflix-muted);
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ---------- NAV BUTTONS ---------- */
.stButton > button {
    background-color: transparent !important;
    color: #e5e5e5 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

.stButton > button:hover {
    color: white !important;
    background-color: rgba(255,255,255,0.08) !important;
}

/* Active nav feel via key styling is limited; we use red accent on primary actions */
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background-color: var(--netflix-red) !important;
    color: white !important;
}

div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover {
    background-color: var(--netflix-red-hover) !important;
}

/* ---------- HERO ---------- */
.hero-banner {
    position: relative;
    background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 40%, #141414 100%);
    border-radius: 12px;
    padding: 48px 40px;
    margin-bottom: 32px;
    overflow: hidden;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 55%;
    background: linear-gradient(90deg, transparent 0%, rgba(20,20,20,0.3) 40%, rgba(20,20,20,0.85) 100%);
    pointer-events: none;
}

.hero-label {
    font-size: 13px;
    font-weight: 700;
    color: var(--netflix-red);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
    color: white;
    margin: 0 0 12px 0;
    line-height: 1.1;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 16px;
    color: rgba(255,255,255,0.85);
    max-width: 480px;
    line-height: 1.5;
    margin-bottom: 24px;
}

/* ---------- SECTION TITLES ---------- */
.section-title {
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin: 36px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title .material-icons-round {
    color: var(--netflix-red);
}

/* ---------- HORIZONTAL SHELF ---------- */
.shelf {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding-bottom: 12px;
    margin-bottom: 8px;
    scrollbar-width: thin;
    scrollbar-color: #333 transparent;
}

.shelf::-webkit-scrollbar {
    height: 6px;
}
.shelf::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 3px;
}

.shelf-card {
    flex: 0 0 160px;
    background: var(--netflix-card);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: pointer;
}

.shelf-card:hover {
    transform: scale(1.06);
    box-shadow: 0 12px 28px rgba(229,9,20,0.35);
}

.shelf-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    background: #1a1a1a;
    display: block;
}

.shelf-info {
    padding: 10px 12px 14px;
}

.shelf-title {
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin: 0 0 4px 0;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.shelf-meta {
    font-size: 12px;
    color: var(--netflix-red);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ---------- MOVIE OF THE DAY ---------- */
.motd-banner {
    background: linear-gradient(135deg, #1f1f1f 0%, #141414 100%);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 28px;
    display: flex;
    gap: 28px;
    align-items: flex-start;
    border-left: 5px solid var(--netflix-red);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

.motd-poster {
    width: 140px;
    min-width: 140px;
    height: 210px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.6);
}

.motd-label {
    font-size: 12px;
    color: var(--netflix-red);
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.motd-title {
    font-size: 28px;
    font-weight: 900;
    color: white;
    margin: 0 0 10px 0;
    line-height: 1.15;
}

.motd-overview {
    font-size: 14px;
    color: var(--netflix-muted);
    line-height: 1.6;
}

/* ---------- GENRE GRID ---------- */
.genre-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 12px;
}

.genre-card {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    height: 140px;
    background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
    border: 1px solid #2a2a2a;
    transition: all 0.25s ease;
    cursor: pointer;
}

.genre-card:hover {
    transform: translateY(-4px);
    border-color: var(--netflix-red);
    box-shadow: 0 12px 30px rgba(229,9,20,0.3);
}

.genre-card-content {
    position: absolute;
    inset: 0;
    padding: 20px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.3) 60%, transparent 100%);
}

.genre-name {
    font-size: 20px;
    font-weight: 800;
    color: white;
    margin: 0;
}

.genre-count {
    font-size: 13px;
    color: var(--netflix-muted);
    margin-top: 4px;
}

.genre-icon {
    position: absolute;
    top: 16px;
    right: 16px;
    font-size: 28px !important;
    color: rgba(255,255,255,0.25);
}

/* ---------- SEARCH / RESULT CARDS (GRID) ---------- */
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 16px;
}

.result-card {
    background: var(--netflix-card);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2a2a2a;
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
}

.result-card:hover {
    border-color: var(--netflix-red);
    box-shadow: 0 10px 28px rgba(229,9,20,0.25);
    transform: translateY(-3px);
}

.result-poster {
    width: 100%;
    aspect-ratio: 16/9;
    object-fit: cover;
    background: #1a1a1a;
}

.result-body {
    padding: 16px;
    flex: 1;
    display: flex;
    flex-direction: column;
}

.result-title {
    font-size: 16px;
    font-weight: 800;
    color: white;
    margin: 0 0 8px 0;
    line-height: 1.3;
}

.result-meta {
    display: flex;
    gap: 12px;
    font-size: 13px;
    color: var(--netflix-red);
    font-weight: 700;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.result-overview {
    font-size: 13px;
    color: var(--netflix-muted);
    line-height: 1.5;
    flex: 1;
}

/* ---------- WATCHLIST POSTER GRID ---------- */
.watch-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 18px;
    margin-top: 16px;
}

.watch-card {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    background: var(--netflix-card);
    transition: transform 0.25s ease;
}

.watch-card:hover {
    transform: scale(1.04);
}

.watch-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}

.watch-info {
    padding: 10px 12px 14px;
}

.watch-title {
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin: 0 0 4px 0;
    line-height: 1.3;
}

.watch-rating {
    font-size: 12px;
    color: var(--netflix-red);
    font-weight: 600;
}

/* ---------- INPUTS ---------- */
.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--netflix-red) !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.3) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: white !important;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    padding: 48px 0 20px;
    color: #555;
    font-size: 12px;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 768px) {
    .hero-title { font-size: 28px; }
    .hero-banner { padding: 32px 20px; min-height: 220px; }
    .motd-banner { flex-direction: column; }
    .motd-poster { width: 100%; height: auto; max-height: 280px; }
    .shelf-card { flex: 0 0 130px; }
    .result-grid { grid-template-columns: 1fr; }
    .genre-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
}
</style>
"""

st.markdown(NETFLIX_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def icon(name: str, size: int = 22, color: str = "#E50914") -> str:
    return (
        f'<span class="material-icons-round" '
        f'style="font-size:{size}px;color:{color};vertical-align:middle;">'
        f'{name}</span>'
    )


def make_poster_placeholder(title: str, height: str = "100%") -> str:
    short = (title[:22] + "…") if len(title) > 22 else title
    return (
        f'<div style="width:100%;height:{height};min-height:180px;'
        f'background:linear-gradient(135deg,#E50914,#7a0009);'
        f'display:flex;flex-direction:column;align-items:center;'
        f'justify-content:center;color:white;text-align:center;padding:12px;">'
        f'<span class="material-icons-round" style="font-size:36px;opacity:0.9;">movie</span>'
        f'<span style="margin-top:8px;font-size:12px;font-weight:700;line-height:1.2;">{short}</span>'
        f'</div>'
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
                "poster": f"{TMDB_IMG_BASE}{m['poster_path']}" if m.get("poster_path") else None,
                "rating": m.get("vote_average", 0),
                "release": (m.get("release_date") or "")[:4],
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
        pass
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
        pass
    return None


# ============================================================
# DATA + NLP ENGINE
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


def _parse_names(text: str) -> str:
    try:
        items = ast.literal_eval(text)
        return " ".join(i["name"] for i in items).lower()
    except Exception:
        return ""


def _clean(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


@st.cache_resource(show_spinner="Loading CineMatch engine…")
def build_engine():
    df = load_data().copy()
    df = df.dropna(subset=["title", "overview"])
    df["genres"] = df["genres"].apply(_parse_names)
    df["keywords"] = df["keywords"].apply(_parse_names)
    df["overview"] = df["overview"].apply(_clean)
    df["soup"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

    keep = [c for c in ["id", "title", "overview", "genres", "vote_average", "release_date", "soup"] if c in df.columns]
    df = df[keep].reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=2)
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
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1 : (n * 5) + 1]
    indices = [i for i, _ in scores]
    result = df.iloc[indices][["id", "title", "overview", "genres", "vote_average"]].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    if genre_filter:
        mask = result["genres"].apply(lambda g: any(gen.lower() in g for gen in genre_filter))
        result = result[mask]
    return result.head(n)


def search_titles(df, query, limit=5):
    query = query.lower()
    return df[df["title"].str.lower().str.contains(query, na=False)]["title"].head(limit).tolist()


# ============================================================
# GENRE HELPERS
# ============================================================
GENRE_ICONS = {
    "Action": "flash_on",
    "Adventure": "explore",
    "Animation": "animation",
    "Comedy": "sentiment_very_satisfied",
    "Crime": "gavel",
    "Documentary": "videocam",
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

GENRE_LABELS = {"Science Fiction": "Sci-Fi"}

BROWSE_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western",
]


def get_genre_movies(df, genre_name, n=20):
    genre_key = genre_name.lower()
    mask = df["genres"].str.contains(genre_key, na=False)
    filtered = df[mask].copy()
    if "vote_average" in filtered.columns:
        filtered = filtered.sort_values("vote_average", ascending=False)
    return filtered.head(n)


def count_genre_movies(df, genre_name):
    return int(df["genres"].str.contains(genre_name.lower(), na=False).sum())


# ============================================================
# WATCHLIST HELPERS
# ============================================================
def add_to_watchlist(movie_id, title, poster=None, rating=None):
    for m in st.session_state.watchlist:
        if m["id"] == movie_id:
            return False
    st.session_state.watchlist.append({
        "id": movie_id,
        "title": title,
        "poster": poster,
        "rating": rating,
    })
    return True


def remove_from_watchlist(movie_id):
    st.session_state.watchlist = [m for m in st.session_state.watchlist if m["id"] != movie_id]


def is_in_watchlist(movie_id):
    return any(m["id"] == movie_id for m in st.session_state.watchlist)


# ============================================================
# HEADER + NAV
# ============================================================
st.markdown(
    f"""
    <div class="cinematch-header">
        <div>
            <div class="cinematch-logo">CINEMATCH</div>
            <div class="cinematch-tagline">YOUR NEXT FAVORITE MOVIE IS ONE MATCH AWAY</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    if st.button("HOME", use_container_width=True, key="nav_home",
                 type="primary" if st.session_state.page == "home" else "secondary"):
        st.session_state.page = "home"
        st.session_state.selected_genre = None
        st.rerun()
with nav2:
    if st.button("BROWSE", use_container_width=True, key="nav_browse",
                 type="primary" if st.session_state.page == "browse" else "secondary"):
        st.session_state.page = "browse"
        st.session_state.selected_genre = None
        st.rerun()
with nav3:
    if st.button("SEARCH", use_container_width=True, key="nav_search",
                 type="primary" if st.session_state.page == "search" else "secondary"):
        st.session_state.page = "search"
        st.rerun()
with nav4:
    wl_count = len(st.session_state.watchlist)
    if st.button(f"WATCHLIST ({wl_count})", use_container_width=True, key="nav_watch",
                 type="primary" if st.session_state.page == "watchlist" else "secondary"):
        st.session_state.page = "watchlist"
        st.rerun()


# ============================================================
# LOAD ENGINE
# ============================================================
df, sim = build_engine()


# ============================================================
# RENDER HELPERS
# ============================================================
def render_shelf_card(title, poster, rating, year=None):
    poster_html = (
        f'<img class="shelf-poster" src="{poster}" alt="{title}">'
        if poster else make_poster_placeholder(title)
    )
    year_txt = f" · {year}" if year else ""
    return f"""
    <div class="shelf-card">
        {poster_html}
        <div class="shelf-info">
            <div class="shelf-title">{title}</div>
            <div class="shelf-meta">{icon('star', 13)} {rating:.1f}{year_txt}</div>
        </div>
    </div>
    """


def render_movie_result_card(row, show_similarity=True):
    movie_id = row.get("id")
    rating = row.get("vote_average") or 0
    sim_val = row.get("similarity")
    overview = str(row.get("overview", ""))[:160] + "…"

    poster = None
    if TMDB_API_KEY:
        poster = fetch_poster_by_id(movie_id) or fetch_poster_by_title(row["title"])

    poster_html = (
        f'<img class="result-poster" src="{poster}" alt="{row["title"]}">'
        if poster else make_poster_placeholder(row["title"], "180px")
    )

    meta = f'{icon("star", 14)} {rating:.1f}/10'
    if show_similarity and sim_val is not None:
        pct = int(sim_val * 100)
        meta += f' &nbsp;·&nbsp; {pct}% Match'

    st.markdown(
        f"""
        <div class="result-card">
            {poster_html}
            <div class="result-body">
                <div class="result-title">{row['title']}</div>
                <div class="result-meta">{meta}</div>
                <div class="result-overview">{overview}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if movie_id is not None:
        in_wl = is_in_watchlist(movie_id)
        label = "♥ Remove" if in_wl else "+ Watchlist"
        if st.button(label, key=f"wl_{movie_id}_{row['title'][:20]}", use_container_width=True):
            if in_wl:
                remove_from_watchlist(movie_id)
            else:
                add_to_watchlist(movie_id, row["title"], poster, f"{rating:.1f}")
            st.rerun()


# ============================================================
# PAGE: HOME
# ============================================================
def render_home():
    # Hero
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-label">Welcome to CineMatch</div>
            <h1 class="hero-title">Find Your Next Obsession</h1>
            <p class="hero-subtitle">
                AI-powered recommendations. Endless stories.<br>
                Discover movies that match your mood.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Movie of the Day
    st.markdown(f'<div class="section-title">{icon("today", 24)} Movie of the Day</div>', unsafe_allow_html=True)

    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top_rated = df[df["vote_average"] >= 7.5] if "vote_average" in df.columns else df
    if not top_rated.empty:
        motd = top_rated.sample(1).iloc[0]
        poster = None
        if TMDB_API_KEY:
            poster = fetch_poster_by_id(motd.get("id")) or fetch_poster_by_title(motd["title"])
        poster_html = (
            f'<img class="motd-poster" src="{poster}" alt="poster">'
            if poster else make_poster_placeholder(motd["title"], "210px")
        )
        rating = motd.get("vote_average", 0)
        st.markdown(
            f"""
            <div class="motd-banner">
                {poster_html}
                <div style="flex:1;min-width:0;">
                    <div class="motd-label">{icon('auto_awesome', 14)} Picked for Today</div>
                    <div class="motd-title">{motd['title']}</div>
                    <div class="result-meta" style="margin-bottom:12px;">
                        {icon('star', 15)} {rating:.1f}/10
                    </div>
                    <div class="motd-overview">{str(motd['overview'])[:380]}…</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trending shelf
    if TMDB_API_KEY:
        st.markdown(
            f'<div class="section-title">{icon("local_fire_department", 24)} Trending This Week</div>',
            unsafe_allow_html=True,
        )
        trending = fetch_trending_movies(12)
        if trending:
            cards = "".join([
                render_shelf_card(m["title"], m["poster"], m["rating"] or 0, m.get("release"))
                for m in trending
            ])
            st.markdown(f'<div class="shelf">{cards}</div>', unsafe_allow_html=True)
        else:
            st.info("Trending unavailable right now.")

    # Explore genres CTA
    st.markdown(
        f'<div class="section-title">{icon("explore", 24)} Or Explore by Genre</div>',
        unsafe_allow_html=True,
    )
    if st.button("BROWSE ALL GENRES →", use_container_width=True, key="home_browse", type="primary"):
        st.session_state.page = "browse"
        st.rerun()


# ============================================================
# PAGE: BROWSE
# ============================================================
def render_browse():
    if st.session_state.selected_genre:
        genre = st.session_state.selected_genre
        if st.button("← Back to all genres", key="back_genres"):
            st.session_state.selected_genre = None
            st.rerun()

        label = GENRE_LABELS.get(genre, genre)
        st.markdown(
            f'<div class="section-title">{icon(GENRE_ICONS.get(genre, "movie"), 24)} {label}</div>',
            unsafe_allow_html=True,
        )

        movies = get_genre_movies(df, genre, n=18)
        if movies.empty:
            st.info(f"No movies found in {label}.")
        else:
            st.caption(f"Top {len(movies)} highest-rated films in {label}")
            # Grid of result cards
            cols = st.columns(3)
            for i, (_, row) in enumerate(movies.iterrows()):
                with cols[i % 3]:
                    render_movie_result_card(row, show_similarity=False)
        return

    # Genre grid
    st.markdown(
        f"""
        <div class="hero-banner" style="min-height:160px;padding:36px 32px;">
            <div class="hero-label">Discover</div>
            <h1 class="hero-title" style="font-size:32px;">Browse by Genre</h1>
            <p class="hero-subtitle" style="margin-bottom:0;">
                Don't know what to watch? Pick a mood and we'll show you the best.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual genre cards
    cols_per_row = 4
    for i in range(0, len(BROWSE_GENRES), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, genre in enumerate(BROWSE_GENRES[i : i + cols_per_row]):
            with cols[j]:
                count = count_genre_movies(df, genre)
                label = GENRE_LABELS.get(genre, genre)
                icon_name = GENRE_ICONS.get(genre, "movie")

                st.markdown(
                    f"""
                    <div class="genre-card">
                        <span class="material-icons-round genre-icon">{icon_name}</span>
                        <div class="genre-card-content">
                            <div class="genre-name">{label}</div>
                            <div class="genre-count">{count:,} movies</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {label}", key=f"genre_{genre}", use_container_width=True):
                    st.session_state.selected_genre = genre
                    st.rerun()


# ============================================================
# PAGE: SEARCH
# ============================================================
def render_search():
    st.markdown(
        f'<div class="section-title">{icon("search", 24)} Search Movies</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        hist = "  ·  ".join(st.session_state.history)
        st.markdown(
            f'<div style="color:#888;font-size:13px;margin-bottom:12px;">'
            f'{icon("history", 15, "#888")} Recent: {hist}</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder="Try: The Dark Knight, Inception, Avatar…",
            label_visibility="collapsed",
            key="search_query",
        )
    with col2:
        n_results = st.selectbox("Show", [6, 9, 12, 15], index=1, label_visibility="collapsed", key="search_n")

    st.markdown(
        f'<div style="color:#888;font-size:13px;margin:12px 0 6px;">'
        f'{icon("tune", 15, "#888")} Optional genre filter</div>',
        unsafe_allow_html=True,
    )
    selected_genres = st.multiselect(
        "Genres",
        options=BROWSE_GENRES,
        default=st.session_state.genre_filter,
        label_visibility="collapsed",
        format_func=lambda g: GENRE_LABELS.get(g, g),
    )
    st.session_state.genre_filter = selected_genres

    if query and len(query) > 1:
        suggestions = search_titles(df, query)
        if suggestions:
            st.markdown(
                f'<div style="color:#888;font-size:13px;margin-top:-4px;">'
                f'{icon("lightbulb", 14, "#888")} Suggestions: {"  ·  ".join(suggestions)}</div>',
                unsafe_allow_html=True,
            )

    if st.button("FIND MY MATCH", use_container_width=True, key="search_btn", type="primary"):
        if not query:
            st.warning("Please enter a movie title.")
        else:
            clean_q = query.strip().title()
            if clean_q not in st.session_state.history:
                st.session_state.history.insert(0, clean_q)
                st.session_state.history = st.session_state.history[:5]
            st.session_state.last_search = query

            with st.spinner("Finding your matches…"):
                results = recommend(df, sim, query, n=n_results, genre_filter=selected_genres)

            if results.empty:
                st.error(f'No movies found matching “{query}”. Try another title.')
            else:
                st.markdown(
                    f'<div class="section-title">{icon("auto_awesome", 24)} '
                    f'Because you liked “{query.title()}”</div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(3)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 3]:
                        render_movie_result_card(row, show_similarity=True)


# ============================================================
# PAGE: WATCHLIST
# ============================================================
def render_watchlist():
    st.markdown(
        f'<div class="section-title">{icon("favorite", 24)} My Watchlist</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.watchlist:
        st.markdown(
            f"""
            <div style="padding:48px 24px;border-radius:12px;background:#1a1a1a;
                        border:1px solid #2a2a2a;text-align:center;color:#888;">
                <span class="material-icons-round" style="font-size:48px;color:#444;">movie_filter</span>
                <p style="margin-top:16px;font-size:15px;">
                    Your watchlist is empty.<br>
                    Search for movies and tap <strong>+ Watchlist</strong> to save them here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(st.session_state.watchlist)} titles saved")

    # Poster wall
    cols = st.columns(5)
    for i, m in enumerate(st.session_state.watchlist):
        with cols[i % 5]:
            poster = m.get("poster")
            poster_html = (
                f'<img class="watch-poster" src="{poster}" alt="{m["title"]}">'
                if poster else make_poster_placeholder(m["title"])
            )
            st.markdown(
                f"""
                <div class="watch-card">
                    {poster_html}
                    <div class="watch-info">
                        <div class="watch-title">{m['title']}</div>
                        <div class="watch-rating">{icon('star', 12)} {m.get('rating') or 'N/A'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Remove", key=f"rm_{m['id']}", use_container_width=True):
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
        CINEMATCH · Powered by NLP & TF-IDF · Built with {icon('favorite', 13)} at TekHer AI Bootcamp
    </div>
    """,
    unsafe_allow_html=True,
)