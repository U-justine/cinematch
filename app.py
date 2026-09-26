"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
Single-page app with top navigation (no sidebar).
"""

import ast
import datetime
import random
import textwrap
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
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================
for key, default in {
    "page": "browse",
    "watchlist": [],
    "history": [],
    "genre_filter": [],
    "selected_genre": None,
    "last_search": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# RENDER HELPER
# ============================================================
def render_html(markup: str):
    """Render HTML reliably by stripping all leading whitespace."""
    st.html(textwrap.dedent(markup).strip())


# ============================================================
# CSS
# ============================================================
CSS = """
:root {
    --red: #E50914;
    --red-hover: #F40612;
    --black: #0f0f0f;
    --card: #1b1b1b;
    --border: #2a2a2a;
    --white: #FFFFFF;
    --muted: #9a9a9a;
    --gold: #f5c518;
}
html, body, .stApp {
    background-color: var(--black) !important;
    color: var(--white);
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* HEADER */
.cm-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 4px 16px 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
}
.cm-brand { display: flex; align-items: center; gap: 10px; }
.cm-brand-icon {
    width: 34px; height: 34px; background: var(--red);
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0;
}
.cm-brand-text { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }
.cm-brand-text .cine { color: var(--white); }
.cm-brand-text .match { color: var(--red); }

/* NAV */
.nav-bar {
    display: flex; align-items: center; gap: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px; overflow-x: auto;
    scrollbar-width: none;
}
.nav-bar::-webkit-scrollbar { display: none; }
.nav-link {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 14px 16px;
    color: #b3b3b3 !important;
    text-decoration: none !important;
    font-size: 15px; font-weight: 500;
    letter-spacing: 0.3px; position: relative;
    white-space: nowrap; transition: color 0.2s ease;
}
.nav-link:hover { color: #ffffff !important; }
.nav-link.active { color: #ffffff !important; font-weight: 700; }
.nav-link.active::after {
    content: ''; position: absolute;
    left: 12px; right: 12px; bottom: 0;
    height: 2px; background: var(--red);
    border-radius: 2px;
}

/* TITLES */
.page-title { font-size: 34px; font-weight: 900; margin: 0; letter-spacing: -0.5px; }
.page-subtitle { color: var(--muted); font-size: 14.5px; margin: 6px 0 26px 0; }
.page-header {
    display: flex; align-items: baseline; justify-content: space-between;
    flex-wrap: wrap; gap: 10px; margin-bottom: 4px;
}
.page-title-row { display: flex; align-items: center; gap: 12px; }
.hero-title { font-size: 34px; font-weight: 900; margin: 0 0 6px 0; letter-spacing: -0.5px; }
.hero-subtitle { color: var(--muted); font-size: 14.5px; margin-bottom: 26px; }

/* GENRE CARDS */
.genre-card {
    position: relative; height: 168px; border-radius: 10px;
    border: 2px solid var(--red); overflow: hidden;
    background-size: cover; background-position: center;
    margin-bottom: 14px; transition: transform 0.2s ease;
}
.genre-card:hover { transform: translateY(-3px); }
.genre-scrim {
    position: absolute; inset: 0;
    background: linear-gradient(to top,
        rgba(0,0,0,0.92) 15%,
        rgba(0,0,0,0.15) 60%,
        rgba(0,0,0,0.05) 100%);
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 14px 16px;
}
.genre-icon { color: #fff; opacity: 0.95; }
.genre-name { font-size: 20px; font-weight: 800; color: white; margin: 0; }
.genre-count {
    font-size: 13px; color: #e0e0e0; margin-top: 4px;
    display: flex; align-items: center; gap: 6px;
}
.genre-count .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--red); display: inline-block;
}

/* INPUTS */
.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.3) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: white !important;
}

/* RESULT CARDS */
.result-card {
    background: var(--card); border: 2px solid var(--red);
    border-radius: 10px; overflow: hidden;
    margin-bottom: 14px;
}
.result-poster {
    width: 100%; aspect-ratio: 4 / 3; object-fit: cover;
    display: block; background: #111;
}
.result-body { padding: 14px 16px; min-height: 120px; }
.result-title { font-size: 15.5px; font-weight: 800; color: white; margin: 0 0 2px 0; }
.result-year { color: var(--muted); font-size: 12.5px; margin-bottom: 6px; }
.result-meta {
    display: flex; align-items: center; gap: 10px;
    font-size: 12.5px; font-weight: 700; margin-bottom: 8px;
}
.result-meta .rating { color: var(--gold); }
.result-meta .match { color: var(--red); }
.result-overview { font-size: 12px; color: var(--muted); line-height: 1.5; }

/* WATCHLIST */
.watch-count { color: var(--muted); font-size: 14px; margin: 4px 0 24px 0; }
.watch-poster-wrap { position: relative; border-radius: 10px; overflow: hidden; }
.watch-poster {
    width: 100%; aspect-ratio: 2 / 3; object-fit: cover;
    display: block; background: #111;
}
.watch-rating-badge {
    position: absolute; top: 8px; left: 8px;
    background: rgba(20,20,20,0.9); border-radius: 5px;
    padding: 3px 7px; font-size: 12px; font-weight: 800;
    color: var(--gold); display: flex; align-items: center; gap: 3px;
}
.watch-remove-badge {
    position: absolute; top: 8px; right: 8px;
    width: 24px; height: 24px; border-radius: 50%;
    background: var(--red); display: flex;
    align-items: center; justify-content: center;
}
.watch-title { font-size: 14px; font-weight: 800; color: white; margin: 10px 0 2px 0; }
.watch-year { font-size: 12px; color: var(--muted); }

/* EMPTY STATE */
.empty-state {
    padding: 56px 24px; border-radius: 12px;
    background: var(--card); border: 1px solid var(--border);
    text-align: center; color: var(--muted);
}

/* HOME */
.home-hero {
    background: linear-gradient(135deg, #1a0505 0%, #2a0a0a 45%, #0f0f0f 100%);
    border-radius: 12px; padding: 46px 40px; margin-bottom: 30px;
    min-height: 200px; display: flex; flex-direction: column; justify-content: center;
}
.home-hero-label {
    color: var(--red); font-weight: 700; font-size: 12.5px;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
}
.home-hero-title { font-size: 38px; font-weight: 900; margin: 0 0 10px 0; letter-spacing: -0.5px; }
.home-hero-sub { color: #d5d5d5; font-size: 15px; max-width: 480px; line-height: 1.5; }

.section-title {
    font-size: 21px; font-weight: 800; margin: 30px 0 14px 0;
    display: flex; align-items: center; gap: 9px;
}
.shelf { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 10px; }
.shelf-card { flex: 0 0 155px; background: var(--card); border-radius: 8px; overflow: hidden; }
.shelf-poster { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; background: #111; }
.shelf-info { padding: 9px 11px 12px; }
.shelf-title {
    font-size: 12.5px; font-weight: 700; color: white;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 3px;
}
.shelf-meta { font-size: 11.5px; color: var(--gold); font-weight: 700; }

.motd-banner {
    background: var(--card); border-radius: 12px; padding: 26px;
    margin-bottom: 10px; display: flex; gap: 24px;
    border-left: 4px solid var(--red);
}
.motd-poster { width: 130px; min-width: 130px; height: 195px; object-fit: cover; border-radius: 8px; }
.motd-label {
    color: var(--red); font-weight: 800; font-size: 11.5px;
    letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 6px;
}
.motd-title { font-size: 24px; font-weight: 900; margin: 0 0 8px 0; }
.motd-overview { color: var(--muted); font-size: 13.5px; line-height: 1.6; }

/* BUTTONS */
.stButton > button {
    background-color: var(--red) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 18px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    width: 100%;
}
.stButton > button:hover { background-color: var(--red-hover) !important; }

.footer { text-align: center; padding: 40px 0 10px; color: #555; font-size: 12px; }

@media (max-width: 768px) {
    .hero-title, .page-title { font-size: 26px; }
    .home-hero { padding: 32px 20px; }
    .home-hero-title { font-size: 28px; }
    .motd-banner { flex-direction: column; }
    .motd-poster { width: 100%; max-width: 260px; height: auto; }
    .shelf-card { flex: 0 0 130px; }
    .genre-card { height: 140px; }
    .nav-link { font-size: 13px; padding: 12px 10px; }
}
@media (max-width: 480px) {
    .hero-title, .page-title { font-size: 22px; }
    .nav-link { font-size: 11px; padding: 10px 8px; gap: 5px; }
}
"""

render_html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>{CSS}</style>
""")


# ============================================================
# SVG ICONS
# ============================================================
ICON_PATHS = {
    "home": '<path d="M3 12l9-9 9 9"/><path d="M9 21V9h6v12"/>',
    "search": '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.2" y2="16.2"/>',
    "favorite": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "movie": '<rect x="2" y="3" width="20" height="18" rx="2"/><line x1="7" y1="3" x2="7" y2="21"/><line x1="17" y1="3" x2="17" y2="21"/>',
    "theaters": '<rect x="2" y="3" width="20" height="18" rx="2"/><line x1="7" y1="3" x2="7" y2="21"/><line x1="17" y1="3" x2="17" y2="21"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "close": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "today": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "auto_awesome": '<path d="M12 2l1.6 6.4L20 10l-6.4 1.6L12 18l-1.6-6.4L4 10l6.4-1.6z"/>',
    "local_fire_department": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "explore": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "history": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "lightbulb": '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.5.4.9 1.1.9 1.8V17a1 1 0 0 0 1 1h4.2a1 1 0 0 0 1-1v-.5c0-.7.4-1.4.9-1.8A7 7 0 0 0 12 2z"/>',
    "bolt": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "rocket_launch": '<path d="M12 2c2.2 2 3.3 5.2 3.3 8.3 0 2.2-.5 3.8-1.1 5.4l-2.2 5.3-2.2-5.3c-.6-1.6-1.1-3.2-1.1-5.4C8.7 7.2 9.8 4 12 2z"/><circle cx="12" cy="9.5" r="1.6"/>',
    "dark_mode": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "theater_comedy": '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
    "masks": '<circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
    "fingerprint": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "camera_alt": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
}

FILLED_ICONS = {"star", "favorite", "bolt", "auto_awesome"}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    inner = ICON_PATHS.get(name, '<circle cx="12" cy="12" r="9"/>')
    if name in FILLED_ICONS:
        style = f'fill="{color}" stroke="none"'
    else:
        style = (
            f'fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round"'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" {style} '
        f'style="vertical-align:middle;display:inline-block;">{inner}</svg>'
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
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/w780"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending_movies(limit: int = 12):
    if not TMDB_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/trending/movie/week",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )
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
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("poster_path"):
            return f"{TMDB_IMG_BASE}{data['poster_path']}"
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_by_title(title: str):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "language": "en-US"},
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return f"{TMDB_IMG_BASE}{results[0]['poster_path']}"
    except Exception:
        pass
    return None


GENRE_ID_MAP = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
    "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14,
    "History": 36, "Horror": 27, "Music": 10402, "Mystery": 9648,
    "Romance": 10749, "Science Fiction": 878, "Thriller": 53,
    "War": 10752, "Western": 37,
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_genre_backdrop(genre_name: str):
    if not TMDB_API_KEY:
        return None
    gid = GENRE_ID_MAP.get(genre_name)
    if not gid:
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/discover/movie",
            params={"api_key": TMDB_API_KEY, "with_genres": gid, "sort_by": "popularity.desc"},
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("backdrop_path"):
            return f"{TMDB_BACKDROP_BASE}{results[0]['backdrop_path']}"
    except Exception:
        pass
    return None


def placeholder_bg(seed_text: str) -> str:
    palettes = [
        "#3a0a0a,#120202", "#0a1a3a,#020712", "#1a0a3a,#070212",
        "#0a3a1a,#021207", "#3a1a0a,#120702",
    ]
    idx = sum(ord(c) for c in seed_text) % len(palettes)
    a, b = palettes[idx].split(",")
    return f"linear-gradient(135deg, {a}, {b})"


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


def _parse_names_titlecase(text: str):
    try:
        items = ast.literal_eval(text)
        return [i["name"] for i in items]
    except Exception:
        return []


def _clean(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


@st.cache_resource(show_spinner="Loading CineMatch engine…")
def build_engine():
    df = load_data().copy()
    df = df.dropna(subset=["title", "overview"])
    df["genres_list"] = df["genres"].apply(_parse_names_titlecase)
    df["genres"] = df["genres"].apply(_parse_names)
    df["keywords"] = df["keywords"].apply(_parse_names)
    df["overview"] = df["overview"].apply(_clean)
    df["soup"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

    keep = [c for c in ["id", "title", "overview", "genres", "genres_list", "vote_average", "release_date", "soup"] if c in df.columns]
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
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1: (n * 5) + 1]
    indices = [i for i, _ in scores]
    result = df.iloc[indices][["id", "title", "overview", "genres", "genres_list", "vote_average", "release_date"]].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    if genre_filter:
        mask = result["genres"].apply(lambda g: any(gen.lower() in g for gen in genre_filter))
        result = result[mask]
    return result.head(n)


def search_titles(df, query, limit=5):
    query = query.lower()
    return df[df["title"].str.lower().str.contains(query, na=False)]["title"].head(limit).tolist()


# ============================================================
# GENRE CONFIG
# ============================================================
GENRE_ICONS = {
    "Action": "bolt",
    "Science Fiction": "rocket_launch",
    "Horror": "dark_mode",
    "Romance": "favorite",
    "Comedy": "theater_comedy",
    "Drama": "masks",
    "Thriller": "fingerprint",
    "Fantasy": "auto_awesome",
    "Animation": "movie",
    "Adventure": "explore",
    "Crime": "shield",
    "Documentary": "camera_alt",
}
GENRE_LABELS = {"Science Fiction": "Sci-Fi"}
BROWSE_GENRES = list(GENRE_ICONS.keys())


def get_genre_movies(df, genre_name, n=20):
    mask = df["genres"].str.contains(genre_name.lower(), na=False)
    filtered = df[mask].copy()
    if "vote_average" in filtered.columns:
        filtered = filtered.sort_values("vote_average", ascending=False)
    return filtered.head(n)


def count_genre_movies(df, genre_name):
    return int(df["genres"].str.contains(genre_name.lower(), na=False).sum())


# ============================================================
# WATCHLIST HELPERS
# ============================================================
def add_to_watchlist(movie_id, title, poster=None, rating=None, year=None, genres=None):
    for m in st.session_state.watchlist:
        if m["id"] == movie_id:
            return False
    st.session_state.watchlist.insert(0, {
        "id": movie_id, "title": title, "poster": poster,
        "rating": rating, "year": year, "genres": genres or [],
    })
    return True


def remove_from_watchlist(movie_id):
    st.session_state.watchlist = [m for m in st.session_state.watchlist if m["id"] != movie_id]


def is_in_watchlist(movie_id):
    return any(m["id"] == movie_id for m in st.session_state.watchlist)


# ============================================================
# HEADER + NAV
# ============================================================
render_html(f"""
<div class="cm-header">
    <div class="cm-brand">
        <div class="cm-brand-icon">{icon('theaters', 18, 'white')}</div>
        <div class="cm-brand-text"><span class="cine">Cine</span><span class="match">Match</span></div>
    </div>
</div>
""")

current_page = st.session_state.page
wl_count = len(st.session_state.watchlist)


def nav_link(page_key: str, icon_name: str, label: str) -> str:
    active = "active" if current_page == page_key else ""
    return (
        f'<a class="nav-link {active}" href="?page={page_key}" target="_self">'
        f'{icon(icon_name, 18, "currentColor")}<span>{label}</span>'
        f'</a>'
    )


render_html(f"""
<div class="nav-bar">
    {nav_link("home", "home", "Home")}
    {nav_link("browse", "search", "Browse")}
    {nav_link("search", "search", "Search")}
    {nav_link("watchlist", "favorite", f"My List ({wl_count})")}
</div>
""")


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
        if poster else
        f'<div class="shelf-poster" style="background:{placeholder_bg(title)};'
        f'display:flex;align-items:center;justify-content:center;">{icon("movie", 30, "rgba(255,255,255,0.5)")}</div>'
    )
    year_txt = f" · {year}" if year else ""
    return f"""
    <div class="shelf-card">
        {poster_html}
        <div class="shelf-info">
            <div class="shelf-title">{title}</div>
            <div class="shelf-meta">{icon('star', 12, 'var(--gold)')} {rating:.1f}{year_txt}</div>
        </div>
    </div>
    """


def render_result_card(row, show_similarity=True):
    movie_id = row.get("id")
    rating = row.get("vote_average") or 0
    sim_val = row.get("similarity")
    year = str(row.get("release_date") or "")[:4]
    overview = str(row.get("overview", ""))[:130] + "…"

    poster = None
    if TMDB_API_KEY:
        poster = fetch_poster_by_id(movie_id) or fetch_poster_by_title(row["title"])

    poster_html = (
        f'<img class="result-poster" src="{poster}" alt="{row["title"]}">'
        if poster else
        f'<div class="result-poster" style="background:{placeholder_bg(row["title"])};'
        f'display:flex;align-items:center;justify-content:center;">{icon("movie", 34, "rgba(255,255,255,0.5)")}</div>'
    )

    match_html = ""
    if show_similarity and sim_val is not None:
        pct = int(sim_val * 100)
        match_html = f'<span class="match">{pct}% Similar</span>'

    render_html(f"""
        <div class="result-card">
            {poster_html}
            <div class="result-body">
                <div class="result-title">{row['title']}</div>
                <div class="result-year">{year}</div>
                <div class="result-meta">
                    <span class="rating">{icon('star', 13, 'var(--gold)')} {rating:.1f}</span>
                    {match_html}
                </div>
                <div class="result-overview">{overview}</div>
            </div>
        </div>
    """)

    if movie_id is not None:
        in_wl = is_in_watchlist(movie_id)
        label = "In Watchlist — tap to remove" if in_wl else "Add to Watchlist"
        if st.button(label, key=f"wl_{movie_id}_{row['title'][:20]}", use_container_width=True):
            if in_wl:
                remove_from_watchlist(movie_id)
            else:
                add_to_watchlist(movie_id, row["title"], poster, f"{rating:.1f}", year, row.get("genres_list"))
            st.rerun()


# ============================================================
# PAGE: HOME
# ============================================================
def render_home():
    render_html("""
        <div class="home-hero">
            <div class="home-hero-label">Welcome to CineMatch</div>
            <h1 class="home-hero-title">Find Your Next Obsession</h1>
            <p class="home-hero-sub">AI-powered recommendations. Endless stories. Discover movies that match your mood.</p>
        </div>
    """)

    render_html(f'<div class="section-title">{icon("today", 20, "var(--red)")} Movie of the Day</div>')

    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top = df[df["vote_average"] >= 7.5] if "vote_average" in df.columns else df
    if not top.empty:
        motd = top.sample(1).iloc[0]
        poster = None
        if TMDB_API_KEY:
            poster = fetch_poster_by_id(motd.get("id")) or fetch_poster_by_title(motd["title"])
        poster_html = (
            f'<img class="motd-poster" src="{poster}" alt="poster">'
            if poster else
            f'<div class="motd-poster" style="background:{placeholder_bg(motd["title"])};'
            f'display:flex;align-items:center;justify-content:center;">{icon("movie", 32, "rgba(255,255,255,0.5)")}</div>'
        )
        rating = motd.get("vote_average", 0)
        render_html(f"""
            <div class="motd-banner">
                {poster_html}
                <div style="flex:1;min-width:0;">
                    <div class="motd-label">{icon('auto_awesome', 13, 'var(--red)')} Picked for Today</div>
                    <div class="motd-title">{motd['title']}</div>
                    <div class="result-meta" style="margin-bottom:10px;">
                        <span class="rating">{icon('star', 14, 'var(--gold)')} {rating:.1f}/10</span>
                    </div>
                    <div class="motd-overview">{str(motd['overview'])[:340]}…</div>
                </div>
            </div>
        """)

    if TMDB_API_KEY:
        render_html(f'<div class="section-title">{icon("local_fire_department", 20, "var(--red)")} Trending This Week</div>')
        trending = fetch_trending_movies(12)
        if trending:
            cards = "".join([
                render_shelf_card(m["title"], m["poster"], m["rating"] or 0, m.get("release"))
                for m in trending
            ])
            render_html(f'<div class="shelf">{cards}</div>')
        else:
            st.info("Trending unavailable right now.")

    render_html(f'<div class="section-title">{icon("explore", 20, "var(--red)")} Or Explore by Genre</div>')
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
        render_html(f'<h1 class="page-title">{label}</h1>')

        movies = get_genre_movies(df, genre, n=20)
        if movies.empty:
            st.info(f"No movies found in {label}.")
        else:
            render_html(f'<p class="page-subtitle">Top {len(movies)} highest-rated films in {label}</p>')
            cols = st.columns(4)
            for i, (_, row) in enumerate(movies.iterrows()):
                with cols[i % 4]:
                    render_result_card(row, show_similarity=False)
        return

    render_html('<h1 class="hero-title">Browse by Genre</h1>')
    render_html('<p class="hero-subtitle">Explore movies by genre. Discover stories that match your mood.</p>')

    cols_per_row = 4
    for i in range(0, len(BROWSE_GENRES), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, genre in enumerate(BROWSE_GENRES[i: i + cols_per_row]):
            with cols[j]:
                count = count_genre_movies(df, genre)
                label = GENRE_LABELS.get(genre, genre)
                icon_name = GENRE_ICONS.get(genre, "movie")

                backdrop = fetch_genre_backdrop(genre) if TMDB_API_KEY else None
                bg_style = (
                    f"background-image: url('{backdrop}');"
                    if backdrop else
                    f"background-image: {placeholder_bg(genre)};"
                )

                render_html(f"""
                    <div class="genre-card" style="{bg_style}">
                        <div class="genre-scrim">
                            <div class="genre-icon">{icon(icon_name, 30, "white")}</div>
                            <div>
                                <div class="genre-name">{label}</div>
                                <div class="genre-count"><span class="dot"></span>{count:,} movies</div>
                            </div>
                        </div>
                    </div>
                """)
                if st.button(f"Open {label}", key=f"genre_{genre}", use_container_width=True):
                    st.session_state.selected_genre = genre
                    st.rerun()


# ============================================================
# PAGE: SEARCH
# ============================================================
def render_search():
    col_input, col_btn = st.columns([9, 1])
    with col_input:
        query = st.text_input(
            "Search", placeholder="Try: The Dark Knight, Inception, Avatar…",
            label_visibility="collapsed", key="search_query",
        )
    with col_btn:
        go = st.button("Search", key="search_go_btn", use_container_width=True, type="primary")

    if st.session_state.history:
        hist = "  ·  ".join(st.session_state.history)
        render_html(f'<div style="color:#888;font-size:12.5px;margin:-6px 0 18px 2px;">'
                    f'{icon("history", 14, "#888")} Recent: {hist}</div>')

    with st.expander("Filter by genre (optional)"):
        selected_genres = st.multiselect(
            "Genres", options=BROWSE_GENRES, default=st.session_state.genre_filter,
            label_visibility="collapsed", format_func=lambda g: GENRE_LABELS.get(g, g),
        )
        st.session_state.genre_filter = selected_genres

    if query and len(query) > 1:
        suggestions = search_titles(df, query)
        if suggestions:
            render_html(f'<div style="color:#888;font-size:12.5px;margin:2px 0 18px 2px;">'
                        f'{icon("lightbulb", 14, "#888")} Suggestions: {"  ·  ".join(suggestions)}</div>')

    if go:
        if not query:
            st.warning("Please enter a movie title.")
        else:
            clean_q = query.strip().title()
            if clean_q not in st.session_state.history:
                st.session_state.history.insert(0, clean_q)
                st.session_state.history = st.session_state.history[:5]
            st.session_state.last_search = query

            with st.spinner("Finding your matches…"):
                results = recommend(df, sim, query, n=8, genre_filter=st.session_state.genre_filter)

            if results.empty:
                st.error(f'No movies found matching "{query}". Try another title.')
            else:
                render_html(f'<div style="font-size:24px;font-weight:800;margin:6px 0 22px 0;">Because you liked <em style="color:var(--red);font-style:italic;border-bottom:2px solid var(--red);padding-bottom:2px;">"{query.title()}"</em></div>')
                cols = st.columns(4)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 4]:
                        render_result_card(row, show_similarity=True)


# ============================================================
# PAGE: WATCHLIST
# ============================================================
def render_watchlist():
    render_html(f"""
        <div class="page-header">
            <div class="page-title-row">
                <h1 class="page-title">My Watchlist</h1>
                {icon('favorite', 26, 'var(--red)')}
            </div>
        </div>
    """)

    n = len(st.session_state.watchlist)
    render_html(f'<div class="watch-count">{n} title{"s" if n != 1 else ""} · Sorted by added date</div>')

    if not st.session_state.watchlist:
        render_html(f"""
            <div class="empty-state">
                {icon('movie', 44, '#444')}
                <p style="margin-top:16px;font-size:15px;">
                    Your watchlist is empty.<br>
                    Search for movies and tap <strong>Add to Watchlist</strong> to save them here.
                </p>
            </div>
        """)
        return

    cols = st.columns(5)
    for i, m in enumerate(st.session_state.watchlist):
        with cols[i % 5]:
            poster = m.get("poster")
            poster_html = (
                f'<img class="watch-poster" src="{poster}" alt="{m["title"]}">'
                if poster else
                f'<div class="watch-poster" style="background:{placeholder_bg(m["title"])};'
                f'display:flex;align-items:center;justify-content:center;">{icon("movie", 28, "rgba(255,255,255,0.5)")}</div>'
            )
            genres_txt = ", ".join((m.get("genres") or [])[:2])
            year_genres = " · ".join([x for x in [m.get("year"), genres_txt] if x])

            render_html(f"""
                <div class="watch-card">
                    <div class="watch-poster-wrap">
                        {poster_html}
                        <div class="watch-rating-badge">{icon('star', 11, 'var(--gold)')} {m.get('rating') or 'N/A'}</div>
                        <div class="watch-remove-badge">{icon('close', 12, 'white')}</div>
                    </div>
                    <div class="watch-title">{m['title']}</div>
                    <div class="watch-year">{year_genres}</div>
                </div>
            """)
            if st.button("Remove", key=f"rm_{m['id']}", use_container_width=True):
                remove_from_watchlist(m["id"])
                st.rerun()


# ============================================================
# ROUTER
# ============================================================
PAGES = {
    "home": render_home,
    "browse": render_browse,
    "search": render_search,
    "watchlist": render_watchlist,
}
PAGES.get(st.session_state.page, render_browse)()


# ============================================================
# FOOTER
# ============================================================
render_html(f"""
    <div class="footer">
        CINEMATCH · Powered by NLP & TF-IDF · Built with {icon('favorite', 12, '#555')} by Justine Umutoni © 2026
    </div>
""")