"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TDb API integration.
Uses unicode glyphs + st.html() for reliable rendering.
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
    initial_sidebar_state="expanded",
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
    "wl_sort": "Added (Newest)",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# UNICODE GLYPHS (no font loading needed)
# ============================================================
GLYPHS = {
    "home": "⌂",
    "browse": "▦",
    "search": "⌕",
    "favorite": "♥",
    "star": "★",
    "collections": "❏",
    "profile": "◉",
    "logout": "⏻",
    "movie": "▣",
    "today": "◈",
    "fire": "▲",
    "explore": "✦",
    "bolt": "⚡",
    "rocket": "▲",
    "dark_mode": "☾",
    "comedy": "☺",
    "masks": "☺",
    "fingerprint": "❋",
    "auto": "✦",
    "shield": "◈",
    "camera": "▷",
    "heart": "♥",
    "close": "✕",
    "history": "◔",
    "lightbulb": "◐",
    "theaters": "▣",
}


def g(key: str, size: int = 20, color: str = "#E50914") -> str:
    """Render a unicode glyph — no external font, always works."""
    glyph = GLYPHS.get(key, "●")
    return (
        f'<span style="font-size:{size}px;color:{color};'
        f'vertical-align:middle;font-family:Arial,sans-serif;'
        f'line-height:1;display:inline-block;">{glyph}</span>'
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
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP = "https://image.tmdb.org/t/p/w780"


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
                "poster": f"{TMDB_IMG}{m['poster_path']}" if m.get("poster_path") else None,
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
            return f"{TMDB_IMG}{data['poster_path']}"
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
            return f"{TMDB_IMG}{results[0]['poster_path']}"
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
            return f"{TMDB_BACKDROP}{results[0]['backdrop_path']}"
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
        st.error(f"Could not load dataset: {e}")
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
GENRE_GLYPHS = {
    "Action": "bolt",
    "Science Fiction": "rocket",
    "Horror": "dark_mode",
    "Romance": "favorite",
    "Comedy": "comedy",
    "Drama": "masks",
    "Thriller": "fingerprint",
    "Fantasy": "auto",
    "Animation": "movie",
    "Adventure": "explore",
    "Crime": "shield",
    "Documentary": "camera",
}
GENRE_LABELS = {"Science Fiction": "Sci-Fi"}
BROWSE_GENRES = list(GENRE_GLYPHS.keys())


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
# CSS — using st.html() (renders reliably)
# ============================================================
st.html("""
<style>
:root {
    --red: #E50914;
    --red-hover: #F40612;
    --black: #0f0f0f;
    --panel: #141414;
    --card: #1b1b1b;
    --card-2: #1f1f1f;
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
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--border);
    min-width: 250px !important;
    max-width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0 4px 20px 4px; margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.sidebar-logo-icon {
    width: 34px; height: 34px; background: var(--red);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; color: white; font-size: 20px; font-weight: 900;
}
.sidebar-logo-text { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; }
.sidebar-logo-text .cine { color: var(--white); }
.sidebar-logo-text .match { color: var(--red); }
.sidebar-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 10px 4px 10px 4px;
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #cfcfcf !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 12px !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    width: 100%; box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button p { text-align: left !important; }
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.06) !important;
    color: white !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: rgba(229,9,20,0.12) !important;
    color: var(--red) !important;
    border-left: 3px solid var(--red) !important;
    border-radius: 6px !important;
}

/* Page headers */
.page-title {
    font-size: 34px; font-weight: 900;
    margin: 0; letter-spacing: -0.5px;
}
.page-subtitle {
    color: var(--muted); font-size: 14.5px;
    margin: 6px 0 26px 0;
}
.hero-title { font-size: 34px; font-weight: 900; margin: 0 0 6px 0; }
.hero-subtitle { color: var(--muted); font-size: 14.5px; margin-bottom: 26px; }

/* Genre cards */
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
        rgba(0,0,0,0.92) 15%, rgba(0,0,0,0.15) 60%, rgba(0,0,0,0.05) 100%);
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 14px 16px;
}
.genre-icon { color: #fff; opacity: 0.95; font-size: 28px; }
.genre-name { font-size: 20px; font-weight: 800; color: white; margin: 0; }
.genre-count {
    font-size: 13px; color: #e0e0e0; margin-top: 4px;
    display: flex; align-items: center; gap: 6px;
}
.genre-count .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--red); display: inline-block;
}

/* Search input */
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

/* Result cards */
.result-card {
    background: var(--card); border: 2px solid var(--red);
    border-radius: 10px; overflow: hidden; margin-bottom: 14px;
}
.result-poster {
    width: 100%; aspect-ratio: 4 / 3; object-fit: cover;
    display: block; background: #111;
}
.result-body { padding: 14px 16px; min-height: 128px; }
.result-title { font-size: 15.5px; font-weight: 800; color: white; margin: 0 0 2px 0; }
.result-year { color: var(--muted); font-size: 12.5px; margin-bottom: 6px; }
.result-meta {
    display: flex; align-items: center; gap: 10px;
    font-size: 12.5px; font-weight: 700; margin-bottom: 8px;
}
.result-meta .rating { color: var(--gold); }
.result-meta .match { color: var(--red); }
.result-overview { font-size: 12px; color: var(--muted); line-height: 1.5; }

/* Watchlist */
.watch-count { color: var(--muted); font-size: 14px; margin: 4px 0 24px 0; }
.watch-poster {
    width: 100%; aspect-ratio: 2 / 3; object-fit: cover;
    display: block; background: #111; border-radius: 10px;
}
.watch-title { font-size: 14px; font-weight: 800; color: white; margin: 10px 0 2px 0; }
.watch-year { font-size: 12px; color: var(--muted); }

/* Empty state */
.empty-state {
    padding: 56px 24px; border-radius: 12px; background: var(--card);
    border: 1px solid var(--border); text-align: center; color: var(--muted);
}

/* Home hero */
.home-hero {
    background: linear-gradient(135deg, #1a0505 0%, #2a0a0a 45%, #0f0f0f 100%);
    border-radius: 12px; padding: 46px 40px; margin-bottom: 30px;
    min-height: 200px; display: flex; flex-direction: column; justify-content: center;
}
.home-hero-label {
    color: var(--red); font-weight: 700; font-size: 12.5px;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
}
.home-hero-title {
    font-size: 38px; font-weight: 900; margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}
.home-hero-sub {
    color: #d5d5d5; font-size: 15px;
    max-width: 480px; line-height: 1.5;
}
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
.motd-title { font-size: 24px; font-weight: 900; margin: 0 0 8px 0; }
.motd-overview { color: var(--muted); font-size: 13.5px; line-height: 1.6; }
.motd-label {
    color: var(--red); font-weight: 800; font-size: 11.5px;
    letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 6px;
}

/* Buttons */
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

/* Responsive */
@media (max-width: 768px) {
    .hero-title, .page-title { font-size: 26px; }
    .home-hero { padding: 32px 20px; }
    .home-hero-title { font-size: 28px; }
    .motd-banner { flex-direction: column; }
    .motd-poster { width: 100%; max-width: 260px; height: auto; }
    .shelf-card { flex: 0 0 130px; }
    .genre-card { height: 140px; }
}
</style>
""")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
NAV_ITEMS = [
    ("home", "Home", "home"),
    ("browse", "Browse", "browse"),
    ("watchlist", "My Watchlist", "favorite"),
    ("collections", "Collections", "collections"),
    ("profile", "Profile", "profile"),
]

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-logo">'
        f'<div class="sidebar-logo-icon">▣</div>'
        f'<div class="sidebar-logo-text"><span class="cine">Cine</span><span class="match">Match</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for key, label, glyph_key in NAV_ITEMS:
        is_active = st.session_state.page == key
        cols = st.columns([1, 6])
        with cols[0]:
            color = "var(--red)" if is_active else "#9a9a9a"
            st.markdown(
                f'<div style="padding-top:9px;">{g(glyph_key, 19, color)}</div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button(label, key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.page = key
                st.session_state.selected_genre = None
                st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    scols = st.columns([1, 6])
    with scols[0]:
        st.markdown(f'<div style="padding-top:9px;">{g("search", 19, "#9a9a9a")}</div>', unsafe_allow_html=True)
    with scols[1]:
        if st.button("Search", key="nav_search_link", use_container_width=True,
                     type="primary" if st.session_state.page == "search" else "secondary"):
            st.session_state.page = "search"
            st.rerun()

    lcols = st.columns([1, 6])
    with lcols[0]:
        st.markdown(f'<div style="padding-top:9px;">{g("logout", 19, "#9a9a9a")}</div>', unsafe_allow_html=True)
    with lcols[1]:
        st.button("Log Out", key="nav_logout", use_container_width=True)


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
        f'display:flex;align-items:center;justify-content:center;font-size:30px;color:rgba(255,255,255,0.5);">▣</div>'
    )
    year_txt = f" · {year}" if year else ""
    return (
        f'<div class="shelf-card">'
        f'{poster_html}'
        f'<div class="shelf-info">'
        f'<div class="shelf-title">{title}</div>'
        f'<div class="shelf-meta">{g("star", 11, "#f5c518")} {rating:.1f}{year_txt}</div>'
        f'</div></div>'
    )


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
        f'display:flex;align-items:center;justify-content:center;font-size:34px;color:rgba(255,255,255,0.5);">▣</div>'
    )

    match_html = ""
    if show_similarity and sim_val is not None:
        pct = int(sim_val * 100)
        match_html = f'<span class="match">{pct}% Similar</span>'

    st.markdown(
        f'<div class="result-card">'
        f'{poster_html}'
        f'<div class="result-body">'
        f'<div class="result-title">{row["title"]}</div>'
        f'<div class="result-year">{year}</div>'
        f'<div class="result-meta">'
        f'<span class="rating">{g("star", 12, "#f5c518")} {rating:.1f}</span>'
        f'{match_html}'
        f'</div>'
        f'<div class="result-overview">{overview}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if movie_id is not None:
        in_wl = is_in_watchlist(movie_id)
        label = "♥ In Watchlist — remove" if in_wl else "Add to Watchlist"
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
    st.markdown(
        '<div class="home-hero">'
        '<div class="home-hero-label">Welcome to CineMatch</div>'
        '<h1 class="home-hero-title">Find Your Next Obsession</h1>'
        '<p class="home-hero-sub">AI-powered recommendations. Endless stories. Discover movies that match your mood.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">{g("today", 20)} Movie of the Day</div>',
        unsafe_allow_html=True,
    )

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
            f'display:flex;align-items:center;justify-content:center;font-size:32px;color:rgba(255,255,255,0.5);">▣</div>'
        )
        rating = motd.get("vote_average", 0)
        st.markdown(
            f'<div class="motd-banner">'
            f'{poster_html}'
            f'<div style="flex:1;min-width:0;">'
            f'<div class="motd-label">{g("auto", 13)} Picked for Today</div>'
            f'<div class="motd-title">{motd["title"]}</div>'
            f'<div class="result-meta" style="margin-bottom:10px;">'
            f'<span class="rating">{g("star", 13, "#f5c518")} {rating:.1f}/10</span>'
            f'</div>'
            f'<div class="motd-overview">{str(motd["overview"])[:340]}…</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    if TMDB_API_KEY:
        st.markdown(
            f'<div class="section-title">{g("fire", 20)} Trending This Week</div>',
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
        st.markdown(f'<h1 class="page-title">{label}</h1>', unsafe_allow_html=True)

        movies = get_genre_movies(df, genre, n=20)
        if movies.empty:
            st.info(f"No movies found in {label}.")
        else:
            st.markdown(f'<p class="page-subtitle">Top {len(movies)} highest-rated films in {label}</p>', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, (_, row) in enumerate(movies.iterrows()):
                with cols[i % 4]:
                    render_result_card(row, show_similarity=False)
        return

    st.markdown('<h1 class="hero-title">Browse by Genre</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore movies by genre. Discover stories that match your mood.</p>', unsafe_allow_html=True)

    cols_per_row = 4
    for i in range(0, len(BROWSE_GENRES), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, genre in enumerate(BROWSE_GENRES[i: i + cols_per_row]):
            with cols[j]:
                count = count_genre_movies(df, genre)
                label = GENRE_LABELS.get(genre, genre)
                glyph_key = GENRE_GLYPHS.get(genre, "movie")

                backdrop = fetch_genre_backdrop(genre) if TMDB_API_KEY else None
                bg_style = (
                    f"background-image: url('{backdrop}');"
                    if backdrop else
                    f"background-image: {placeholder_bg(genre)};"
                )

                st.markdown(
                    f'<div class="genre-card" style="{bg_style}">'
                    f'<div class="genre-scrim">'
                    f'<div class="genre-icon">{GLYPHS.get(glyph_key, "●")}</div>'
                    f'<div>'
                    f'<div class="genre-name">{label}</div>'
                    f'<div class="genre-count"><span class="dot"></span>{count:,} movies</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )
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
        go = st.button("Go", key="search_go_btn", use_container_width=True, type="primary")

    if st.session_state.history:
        hist = "  ·  ".join(st.session_state.history)
        st.markdown(
            f'<div style="color:#888;font-size:12.5px;margin:-6px 0 18px 2px;">'
            f'{g("history", 14, "#888")} Recent: {hist}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Filter by genre (optional)"):
        selected_genres = st.multiselect(
            "Genres", options=BROWSE_GENRES, default=st.session_state.genre_filter,
            label_visibility="collapsed", format_func=lambda x: GENRE_LABELS.get(x, x),
        )
        st.session_state.genre_filter = selected_genres

    if query and len(query) > 1:
        suggestions = search_titles(df, query)
        if suggestions:
            st.markdown(
                f'<div style="color:#888;font-size:12.5px;margin:2px 0 18px 2px;">'
                f'{g("lightbulb", 14, "#888")} Suggestions: {"  ·  ".join(suggestions)}</div>',
                unsafe_allow_html=True,
            )

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
                st.markdown(
                    f'<div class="section-title">Because you liked "{query.title()}"</div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(4)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 4]:
                        render_result_card(row, show_similarity=True)


# ============================================================
# PAGE: WATCHLIST
# ============================================================
def render_watchlist():
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title-row">'
        f'<h1 class="page-title">My Watchlist</h1>'
        f'{g("favorite", 26)}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    n = len(st.session_state.watchlist)
    st.markdown(f'<div class="watch-count">{n} title{"s" if n != 1 else ""} · Sorted by added date</div>', unsafe_allow_html=True)

    top_l, top_r = st.columns([6, 2])
    with top_r:
        st.selectbox("Sort", ["Added (Newest)", "Rating (High to Low)", "Title (A–Z)"],
                     label_visibility="collapsed", key="wl_sort")

    if not st.session_state.watchlist:
        st.markdown(
            f'<div class="empty-state">'
            f'{g("movie", 44, "#444")}'
            f'<p style="margin-top:16px;font-size:15px;">'
            f'Your watchlist is empty.<br>Search for movies and tap <strong>Add to Watchlist</strong> to save them here.'
            f'</p></div>',
            unsafe_allow_html=True,
        )
        return

    items = list(st.session_state.watchlist)
    sort_mode = st.session_state.get("wl_sort", "Added (Newest)")
    if sort_mode == "Rating (High to Low)":
        items = sorted(items, key=lambda m: float(m.get("rating") or 0), reverse=True)
    elif sort_mode == "Title (A–Z)":
        items = sorted(items, key=lambda m: m["title"].lower())

    cols = st.columns(5)
    for i, m in enumerate(items):
        with cols[i % 5]:
            poster = m.get("poster")
            poster_html = (
                f'<img class="watch-poster" src="{poster}" alt="{m["title"]}">'
                if poster else
                f'<div class="watch-poster" style="background:{placeholder_bg(m["title"])};'
                f'display:flex;align-items:center;justify-content:center;font-size:28px;color:rgba(255,255,255,0.5);">▣</div>'
            )
            genres_txt = ", ".join((m.get("genres") or [])[:2])
            year_genres = " · ".join([x for x in [m.get("year"), genres_txt] if x])

            st.markdown(
                f'{poster_html}'
                f'<div class="watch-title">{m["title"]}</div>'
                f'<div class="watch-year">{year_genres}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Remove", key=f"rm_{m['id']}", use_container_width=True):
                remove_from_watchlist(m["id"])
                st.rerun()


# ============================================================
# PAGE: COLLECTIONS
# ============================================================
def render_collections():
    st.markdown('<h1 class="page-title">Collections</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Curated groups of titles. Coming soon.</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="empty-state">'
        f'{g("collections", 44, "#444")}'
        f'<p style="margin-top:16px;font-size:15px;">Collections aren\'t set up yet — check back soon.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: PROFILE
# ============================================================
def render_profile():
    st.markdown('<h1 class="page-title">Profile</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Your account details.</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="empty-state">'
        f'{g("profile", 44, "#444")}'
        f'<p style="margin-top:16px;font-size:15px;">Profile settings aren\'t set up yet.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ROUTER
# ============================================================
PAGES = {
    "home": render_home,
    "browse": render_browse,
    "search": render_search,
    "watchlist": render_watchlist,
    "collections": render_collections,
    "profile": render_profile,
}
PAGES.get(st.session_state.page, render_browse)()


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f'<div style="text-align:center;padding:40px 0 10px;color:#555;font-size:12px;">'
    f'CINEMATCH · Powered by NLP & TF-IDF · Built with {g("favorite", 12, "#555")} at TekHer AI Bootcamp'
    f'</div>',
    unsafe_allow_html=True,
)