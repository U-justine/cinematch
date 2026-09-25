"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TDb API integration.
"""

import ast
import html
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
# QUERY PARAM ROUTER
# ============================================================
qp = st.query_params
if "page" in qp:
    st.session_state.page = qp["page"]
elif "page" not in st.session_state:
    st.session_state.page = "home"


# ============================================================
# SESSION STATE
# ============================================================
for key, default in {
    "watchlist": [],
    "history": [],
    "genre_filter": [],
    "selected_genre": None,
    "last_search": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.query_params.get("g") == "none":
    st.session_state.selected_genre = None
if "genre" in st.query_params and st.query_params["genre"]:
    st.session_state.selected_genre = st.query_params["genre"]


# ============================================================
# HELPERS
# ============================================================
def esc(text) -> str:
    """Escape HTML characters so text renders safely."""
    return html.escape(str(text))


def html_block(markup: str):
    """Render HTML correctly — strip ALL leading whitespace so Streamlit doesn't treat it as code."""
    cleaned = textwrap.dedent(markup).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


def icon(name: str, size: int = 22, color: str = "#E50914") -> str:
    return (
        f'<span class="material-icons-round" '
        f'style="font-size:{size}px;color:{color};vertical-align:middle;">'
        f'{name}</span>'
    )


def icon_badge(name: str, size: int = 34, icon_size: int = 18,
               bg: str = "rgba(255,255,255,0.95)", color: str = "#141414") -> str:
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:{bg};display:flex;align-items:center;justify-content:center;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.4);flex-shrink:0;">'
        f'<span class="material-icons-round" style="font-size:{icon_size}px;color:{color};">'
        f'{name}</span></div>'
    )


def make_poster_placeholder(title: str, height: str = "100%", width: str = "100%") -> str:
    short = esc((title[:22] + "…") if len(title) > 22 else title)
    return (
        f'<div style="width:{width};height:{height};min-width:140px;min-height:180px;'
        f'background:linear-gradient(135deg,#E50914,#7a0009);'
        f'display:flex;flex-direction:column;align-items:center;'
        f'justify-content:center;color:white;text-align:center;padding:12px;'
        f'border-radius:8px;flex-shrink:0;">'
        f'<span class="material-icons-round" style="font-size:36px;opacity:0.9;">movie</span>'
        f'<span style="margin-top:8px;font-size:12px;font-weight:700;line-height:1.2;">{short}</span>'
        f'</div>'
    )


# ============================================================
# NETFLIX-STYLE CSS
# ============================================================
NETFLIX_CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons+Round">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap">
<style>
.material-icons-round {
    font-family: 'Material Icons Round' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 24px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}
:root {
    --netflix-red: #E50914;
    --netflix-red-hover: #F40612;
    --netflix-black: #141414;
    --netflix-dark: #181818;
    --netflix-card: #1f1f1f;
    --netflix-muted: #B3B3B3;
    --netflix-white: #FFFFFF;
}
.stApp {
    background-color: var(--netflix-black);
    color: var(--netflix-white);
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}
.block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1500px !important;
}
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

.cinematch-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 12px 8px 12px;
}
.cinematch-logo-row { display: flex; align-items: center; gap: 10px; line-height: 1; }
.cinematch-logo {
    font-size: 34px; font-weight: 900; color: var(--netflix-red);
    letter-spacing: -1.6px; text-transform: uppercase; margin: 0; line-height: 1;
}
.cinematch-logo-icon { font-size: 34px !important; color: var(--netflix-red); }
.header-icons { display: flex; align-items: center; gap: 20px; }

.nav-bar {
    display: flex; align-items: center; gap: 4px;
    padding: 6px 12px 0 12px;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 24px; overflow-x: auto; scrollbar-width: none;
}
.nav-bar::-webkit-scrollbar { display: none; }
.nav-link {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 14px 16px 16px 16px;
    color: #b3b3b3 !important; text-decoration: none !important;
    font-size: 15px; font-weight: 500; letter-spacing: 0.3px;
    position: relative; white-space: nowrap; transition: color 0.2s ease;
}
.nav-link:hover { color: #ffffff !important; }
.nav-link.active { color: #ffffff !important; font-weight: 700; }
.nav-link.active::after {
    content: ''; position: absolute; left: 12px; right: 12px; bottom: 0;
    height: 2px; background: var(--netflix-red); border-radius: 2px;
}
.nav-link .material-icons-round {
    font-size: 20px !important; color: #b3b3b3; transition: color 0.2s ease;
}
.nav-link:hover .material-icons-round,
.nav-link.active .material-icons-round { color: var(--netflix-red) !important; }

.hero-banner {
    background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 40%, #141414 100%);
    border-radius: 12px; padding: 48px 40px; margin-bottom: 32px;
    min-height: 220px; display: flex; flex-direction: column; justify-content: center;
}
.hero-label {
    font-size: 13px; font-weight: 700; color: var(--netflix-red);
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
    font-size: 40px; font-weight: 900; color: white;
    margin: 0 0 10px 0; line-height: 1.1; letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 15px; color: rgba(255,255,255,0.85);
    max-width: 480px; line-height: 1.5;
}

.section-title {
    color: white; font-size: 22px; font-weight: 800;
    margin: 8px 0 16px 0; display: flex; align-items: center; gap: 10px;
}

.shelf { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 12px; margin-bottom: 8px; }
.shelf::-webkit-scrollbar { height: 6px; }
.shelf::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
.shelf-card {
    flex: 0 0 160px; background: var(--netflix-card);
    border-radius: 8px; overflow: hidden; transition: transform 0.25s ease;
}
.shelf-card:hover { transform: scale(1.05); }
.shelf-poster { width: 100%; aspect-ratio: 2/3; object-fit: cover; background: #1a1a1a; display: block; }
.shelf-info { padding: 10px 12px 14px; }
.shelf-title { font-size: 13px; font-weight: 700; color: white; margin: 0 0 4px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.shelf-meta { font-size: 12px; color: var(--netflix-red); font-weight: 600; display: flex; align-items: center; gap: 4px; }

.motd-banner {
    background: linear-gradient(135deg, #1f1f1f 0%, #141414 100%);
    border-radius: 12px; padding: 28px; margin-bottom: 28px;
    display: flex; gap: 28px; align-items: flex-start;
    border-left: 5px solid var(--netflix-red);
}
.motd-banner > div:first-child,
.motd-banner > img:first-child {
    width: 140px !important; min-width: 140px !important;
    max-width: 140px !important; flex-shrink: 0;
}
.motd-poster { width: 140px; height: 210px; object-fit: cover; border-radius: 8px; display: block; }
.motd-label { font-size: 12px; color: var(--netflix-red); font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
.motd-title { font-size: 28px; font-weight: 900; color: white; margin: 0 0 10px 0; }
.motd-overview { font-size: 14px; color: var(--netflix-muted); line-height: 1.6; }

.genre-card {
    position: relative; border-radius: 12px; overflow: hidden; height: 150px;
    border: 2px solid var(--netflix-red); background-size: cover; background-position: center;
    transition: all 0.25s ease; margin-bottom: 8px;
}
.genre-card:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(229,9,20,0.35); }
.genre-card-scrim {
    position: absolute; inset: 0;
    background: linear-gradient(to top, rgba(0,0,0,0.88) 10%, rgba(0,0,0,0.15) 55%, rgba(0,0,0,0.35) 100%);
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 12px 16px 14px;
}
.genre-icon-badge { align-self: flex-start; }
.genre-name { font-size: 19px; font-weight: 800; color: white; margin: 0; }
.genre-count { font-size: 12.5px; color: #e5e5e5; margin-top: 2px; display: flex; align-items: center; gap: 5px; }
.genre-count .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--netflix-red); display: inline-block; }

.result-card {
    position: relative; background: var(--netflix-card); border-radius: 10px;
    overflow: hidden; border: 1px solid #2a2a2a; transition: all 0.25s ease; margin-bottom: 4px;
}
.result-card:hover { border-color: var(--netflix-red); box-shadow: 0 10px 28px rgba(229,9,20,0.25); transform: translateY(-2px); }
.result-poster-wrap { position: relative; }
.result-poster { width: 100%; aspect-ratio: 3/2; object-fit: cover; background: #1a1a1a; display: block; }
.result-body { padding: 14px 16px 16px; }
.result-title { font-size: 15px; font-weight: 800; color: white; margin: 0 0 2px 0; line-height: 1.3; }
.result-year { font-size: 12.5px; color: var(--netflix-muted); margin-bottom: 8px; }
.result-meta { display: flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 700; margin-bottom: 8px; flex-wrap: wrap; }
.result-meta .rating { color: #f5c518; display: flex; align-items: center; gap: 3px; }
.result-meta .match { color: var(--netflix-red); display: flex; align-items: center; gap: 3px; }
.result-overview { font-size: 12.5px; color: var(--netflix-muted); line-height: 1.5; }

.watch-card {
    position: relative; border-radius: 10px; overflow: hidden;
    background: var(--netflix-card); transition: transform 0.2s ease; margin-bottom: 4px;
}
.watch-card:hover { transform: scale(1.03); }
.watch-poster-wrap { position: relative; }
.watch-poster { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.watch-rating-badge {
    position: absolute; top: 8px; left: 8px;
    background: rgba(20,20,20,0.85); border-radius: 6px; padding: 3px 7px;
    font-size: 12px; font-weight: 700; color: #f5c518;
    display: flex; align-items: center; gap: 3px;
}
.watch-info { padding: 10px 12px 4px; }
.watch-title { font-size: 13px; font-weight: 700; color: white; margin: 0 0 2px 0; line-height: 1.3; }
.watch-year { font-size: 11.5px; color: var(--netflix-muted); }

.empty-state {
    padding: 48px 24px; border-radius: 12px; background: #1a1a1a;
    border: 1px solid #2a2a2a; text-align: center; color: #888;
}

.stTextInput > div > div > input {
    background-color: #1a1a1a !important; border: 1px solid #333 !important;
    color: white !important; border-radius: 6px !important;
    padding: 14px 16px !important; font-size: 15px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--netflix-red) !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.3) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background-color: #1a1a1a !important; border: 1px solid #333 !important;
    border-radius: 6px !important; color: white !important;
}
.stButton > button {
    background-color: var(--netflix-red) !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    padding: 10px 18px !important; font-weight: 700 !important;
    font-size: 13.5px !important; letter-spacing: 0.4px !important;
    transition: all 0.2s ease !important; width: 100%;
}
.stButton > button:hover {
    background-color: var(--netflix-red-hover) !important;
    box-shadow: 0 6px 20px rgba(229,9,20,0.5) !important;
}

.footer { text-align: center; padding: 40px 0 20px; color: #555; font-size: 12px; }

@media (max-width: 992px) {
    .cinematch-logo { font-size: 26px; letter-spacing: -1.2px; }
    .cinematch-logo-icon { font-size: 26px !important; }
    .hero-title { font-size: 30px; }
    .section-title { font-size: 19px; }
    .nav-link { font-size: 13px; padding: 12px 12px 14px 12px; }
    .nav-link .material-icons-round { font-size: 18px !important; }
}
@media (max-width: 768px) {
    .cinematch-logo { font-size: 22px; letter-spacing: -1px; }
    .cinematch-logo-icon { font-size: 22px !important; }
    .hero-title { font-size: 26px; }
    .hero-banner { padding: 32px 20px; min-height: 200px; }
    .motd-banner { flex-direction: column; }
    .motd-banner > div:first-child,
    .motd-banner > img:first-child,
    .motd-poster { width: 100% !important; max-width: 280px !important; height: auto !important; }
    .shelf-card { flex: 0 0 130px; }
    .header-icons { gap: 12px; }
    .nav-link { font-size: 12px; padding: 10px 10px 12px 10px; gap: 5px; }
    .nav-link .material-icons-round { font-size: 16px !important; }
}
@media (max-width: 480px) {
    .cinematch-logo { font-size: 18px; letter-spacing: -0.8px; }
    .cinematch-logo-icon { font-size: 18px !important; }
    .nav-link { font-size: 11px; padding: 8px 8px 10px 8px; gap: 4px; }
    .nav-link .material-icons-round { font-size: 14px !important; }
    .header-icons { gap: 8px; }
}
</style>
"""

st.markdown(NETFLIX_CSS, unsafe_allow_html=True)


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
        return None, None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        poster = f"{TMDB_IMG_BASE}{data['poster_path']}" if data.get("poster_path") else None
        backdrop = f"{TMDB_BACKDROP_BASE}{data['backdrop_path']}" if data.get("backdrop_path") else None
        return poster, backdrop
    except Exception:
        return None, None


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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_genre_backdrop(genre_name: str):
    if not TMDB_API_KEY:
        return None
    genre_id_map = {
        "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
        "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14,
        "History": 36, "Horror": 27, "Music": 10402, "Mystery": 9648,
        "Romance": 10749, "Science Fiction": 878, "Thriller": 53,
        "War": 10752, "Western": 37,
    }
    gid = genre_id_map.get(genre_name)
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
    result = df.iloc[indices][["id", "title", "overview", "genres", "vote_average", "release_date"]].copy()
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
    "Action": "flash_on", "Adventure": "explore", "Animation": "child_care",
    "Comedy": "theater_comedy", "Crime": "fingerprint", "Documentary": "camera_alt",
    "Drama": "masks", "Family": "family_restroom", "Fantasy": "auto_awesome",
    "History": "history_edu", "Horror": "sentiment_very_dissatisfied",
    "Music": "music_note", "Mystery": "search", "Romance": "favorite",
    "Science Fiction": "rocket_launch", "Thriller": "bolt",
    "War": "military_tech", "Western": "landscape",
}
GENRE_LABELS = {"Science Fiction": "Sci-Fi"}
BROWSE_GENRES = [
    "Action", "Science Fiction", "Horror", "Romance",
    "Comedy", "Drama", "Thriller", "Fantasy",
    "Animation", "Adventure", "Crime", "Documentary",
]


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
    st.session_state.watchlist.append({
        "id": movie_id, "title": title, "poster": poster,
        "rating": rating, "year": year, "genres": genres,
    })
    return True


def remove_from_watchlist(movie_id):
    st.session_state.watchlist = [m for m in st.session_state.watchlist if m["id"] != movie_id]


def is_in_watchlist(movie_id):
    return any(m["id"] == movie_id for m in st.session_state.watchlist)


# ============================================================
# HEADER + NAV
# ============================================================
current_page = st.session_state.page
wl_count = len(st.session_state.watchlist)


def nav_item(page: str, icon_name: str, label: str) -> str:
    active = "active" if current_page == page else ""
    return (
        f'<a class="nav-link {active}" href="?page={page}" target="_self">'
        f'<span class="material-icons-round">{icon_name}</span>'
        f'<span>{label}</span>'
        f'</a>'
    )


header_html = f"""
<div class="cinematch-header">
    <div class="cinematch-logo-row">
        <span class="material-icons-round cinematch-logo-icon">movie_filter</span>
        <div class="cinematch-logo">CINEMATCH</div>
    </div>
    <div class="header-icons">
        {icon('notifications_none', 24, '#e5e5e5')}
        {icon('account_circle', 26, '#e5e5e5')}
        {icon('expand_more', 18, '#e5e5e5')}
    </div>
</div>
"""

nav_html = f"""
<div class="nav-bar">
    {nav_item('home', 'home', 'HOME')}
    {nav_item('browse', 'grid_view', 'BROWSE')}
    {nav_item('search', 'search', 'SEARCH')}
    {nav_item('watchlist', 'favorite', f'MY LIST ({wl_count})')}
</div>
"""

# Render via html_block so indentation is stripped
html_block(header_html)
html_block(nav_html)


# ============================================================
# LOAD ENGINE
# ============================================================
df, sim = build_engine()


# ============================================================
# RENDER HELPERS
# ============================================================
def render_shelf_card(title, poster, rating, year=None):
    safe_title = esc(title)
    poster_html = (
        f'<img class="shelf-poster" src="{poster}" alt="{safe_title}">'
        if poster else make_poster_placeholder(safe_title)
    )
    year_txt = f" · {esc(year)}" if year else ""
    return f"""
    <div class="shelf-card">
        {poster_html}
        <div class="shelf-info">
            <div class="shelf-title">{safe_title}</div>
            <div class="shelf-meta">{icon('star', 13, '#f5c518')} {rating:.1f}{year_txt}</div>
        </div>
    </div>
    """


def render_movie_result_card(row, show_similarity=True):
    movie_id = row.get("id")
    rating = row.get("vote_average") or 0
    sim_val = row.get("similarity")
    year = str(row.get("release_date") or "")[:4]
    overview = esc(str(row.get("overview", ""))[:150]) + "…"
    safe_title = esc(row["title"])

    poster = None
    if TMDB_API_KEY:
        poster, _ = fetch_poster_by_id(movie_id)
        poster = poster or fetch_poster_by_title(row["title"])

    poster_html = (
        f'<img class="result-poster" src="{poster}" alt="{safe_title}">'
        if poster else make_poster_placeholder(safe_title, "180px")
    )

    match_html = ""
    if show_similarity and sim_val is not None:
        pct = min(99, int(sim_val * 250))
        match_html = f'<span class="match">{icon("bolt", 13, "#E50914")} {pct}% Match</span>'

    card_html = f"""
    <div class="result-card">
        <div class="result-poster-wrap">{poster_html}</div>
        <div class="result-body">
            <div class="result-title">{safe_title}</div>
            <div class="result-year">{esc(year)}</div>
            <div class="result-meta">
                <span class="rating">{icon('star', 13, '#f5c518')} {rating:.1f}</span>
                {match_html}
            </div>
            <div class="result-overview">{overview}</div>
        </div>
    </div>
    """
    html_block(card_html)

    if movie_id is not None:
        in_wl = is_in_watchlist(movie_id)
        label = "Remove from list" if in_wl else "Add to Watchlist"