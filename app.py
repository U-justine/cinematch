"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
Pages: Home, Browse, Search, My List.
No login required.
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
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "history" not in st.session_state:
    st.session_state.history = []
if "genre_filter" not in st.session_state:
    st.session_state.genre_filter = []
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None


# ============================================================
# QUERY PARAM ROUTER
# ============================================================
qp = st.query_params
if "page" in qp:
    st.session_state.page = qp["page"]
elif "page" not in st.session_state:
    st.session_state.page = "home"


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
    --netflix-dark: #1f1f1f;
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
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px !important;
}
.material-icons-round {
    font-family: 'Material Icons Round';
    font-weight: normal; font-style: normal; line-height: 1;
    display: inline-block; vertical-align: middle;
}

/* ---------- TOP NAV BAR ---------- */
.cm-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 12px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}
.cm-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    flex-shrink: 0;
}
.cm-logo {
    font-size: 26px;
    font-weight: 900;
    color: var(--netflix-red);
    letter-spacing: -0.8px;
    text-transform: none;
    margin: 0;
    line-height: 1;
}
.cm-nav {
    display: flex;
    align-items: center;
    gap: 30px;
    flex-wrap: nowrap;
    overflow-x: auto;
    scrollbar-width: none;
}
.cm-nav::-webkit-scrollbar { display: none; }
.cm-nav a {
    color: var(--netflix-muted) !important;
    text-decoration: none !important;
    font-size: 14.5px;
    font-weight: 500;
    letter-spacing: 0.2px;
    white-space: nowrap;
    padding: 6px 0;
    position: relative;
    transition: color 0.2s ease;
}
.cm-nav a:hover { color: var(--netflix-white) !important; }
.cm-nav a.active {
    color: var(--netflix-white) !important;
    font-weight: 700;
}
.cm-nav a.active::after {
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: -14px;
    height: 3px;
    background: var(--netflix-red);
    border-radius: 2px;
}
.cm-icons {
    display: flex;
    align-items: center;
    gap: 18px;
    color: var(--netflix-muted);
    flex-shrink: 0;
}

/* ---------- SEARCH HERO ---------- */
.cm-search-hero {
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border-left: 5px solid var(--netflix-red);
    border-radius: 10px;
    padding: 28px 32px;
    margin-bottom: 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.cm-search-hero .cm-hero-title {
    font-size: 26px;
    font-weight: 900;
    color: var(--netflix-white);
    margin: 0 0 6px 0;
    letter-spacing: -0.4px;
}
.cm-search-hero .cm-hero-sub {
    color: var(--netflix-muted);
    font-size: 14px;
    margin-bottom: 18px;
}

/* ---------- SECTION HEADER ---------- */
.cm-section {
    color: var(--netflix-white);
    font-size: 24px;
    font-weight: 800;
    margin: 34px 0 6px 0;
    letter-spacing: -0.4px;
}
.cm-section-underline {
    width: 60px;
    height: 3px;
    background: var(--netflix-red);
    border-radius: 2px;
    margin-bottom: 20px;
}

/* ---------- MOVIE GRID ---------- */
.cm-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 20px;
    margin-top: 8px;
}
.cm-card {
    background: linear-gradient(180deg, #1f1f1f 0%, #181818 100%);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    display: flex;
    flex-direction: column;
    height: 100%;
}
.cm-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 34px rgba(229,9,20,0.35);
    border-color: var(--netflix-red);
}
.cm-card-poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    background: #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #564d4d;
}
.cm-card-body {
    padding: 14px 16px 16px 16px;
    display: flex;
    flex-direction: column;
    flex: 1;
}
.cm-card-title {
    color: var(--netflix-white);
    font-size: 16px;
    font-weight: 800;
    line-height: 1.25;
    margin: 0 0 6px 0;
    word-break: break-word;
}
.cm-card-meta {
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--netflix-red);
    font-size: 12.5px;
    font-weight: 700;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.cm-card-meta span {
    display: inline-flex; align-items: center; gap: 4px;
}
.cm-card-overview {
    color: var(--netflix-muted);
    font-size: 12.5px;
    line-height: 1.55;
    flex: 1;
    margin-bottom: 12px;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.cm-card-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-top: auto;
}

/* ---------- ACTION BUTTONS ---------- */
.cm-heart {
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.35);
    color: var(--netflix-red);
    border-radius: 50%;
    width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s ease;
}
.cm-heart:hover {
    background: var(--netflix-red);
    color: #fff;
}

/* ---------- GENRE TILES ---------- */
.cm-genre-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 18px;
    margin-top: 10px;
}
.cm-genre-tile {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    min-height: 120px;
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 16px;
}
.cm-genre-tile:hover {
    transform: translateY(-4px);
    border-color: var(--netflix-red);
    box-shadow: 0 10px 28px rgba(229,9,20,0.35);
}
.cm-genre-icon {
    margin-bottom: 8px;
    color: #fff;
    opacity: 0.95;
}
.cm-genre-name {
    color: #fff;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.2px;
    line-height: 1.2;
}
.cm-genre-count {
    color: var(--netflix-muted);
    font-size: 12px;
    margin-top: 3px;
}
.cm-genre-count::before {
    content: '● ';
    color: var(--netflix-red);
}

/* ---------- MISC ---------- */
.cm-history {
    color: #B3B3B3; font-size: 13px;
    margin-bottom: 12px; line-height: 1.8;
}
.cm-empty {
    padding: 40px 20px;
    text-align: center;
    color: var(--netflix-muted);
    background: linear-gradient(135deg, #1f1f1f 0%, #181818 100%);
    border-left: 4px solid var(--netflix-red);
    border-radius: 10px;
}
.cm-footer {
    text-align: center;
    padding: 44px 0 20px 0;
    color: var(--netflix-gray);
    font-size: 12px;
}

/* ---------- STREAMLIT OVERRIDES ---------- */
.stTextInput > div > div > input {
    background-color: rgba(0,0,0,0.75);
    border: 1px solid var(--netflix-gray);
    color: var(--netflix-white) !important;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 15px;
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
    padding: 10px 18px; font-size: 13px; font-weight: 700;
    letter-spacing: 0.4px; transition: all 0.2s ease; width: 100%;
}
.stButton > button:hover {
    background-color: var(--netflix-red-hover);
    box-shadow: 0 6px 20px rgba(229,9,20,0.5);
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 992px) {
    .cm-logo { font-size: 22px; }
    .cm-nav { gap: 22px; }
    .cm-grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
    .cm-section { font-size: 20px; }
    .cm-genre-grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
}
@media (max-width: 640px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .cm-topbar { gap: 12px; padding: 8px 0; }
    .cm-logo { font-size: 20px; }
    .cm-nav { gap: 14px; }
    .cm-nav a { font-size: 12px; }
    .cm-nav a.active::after { bottom: -12px; height: 2.5px; }
    .cm-icons { gap: 10px; }
    .cm-search-hero { padding: 20px 18px; }
    .cm-search-hero .cm-hero-title { font-size: 20px; }
    .cm-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
    .cm-card-title { font-size: 14px; }
    .cm-card-overview { font-size: 11.5px; -webkit-line-clamp: 3; }
    .cm-section { font-size: 18px; }
    .cm-genre-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
    .cm-genre-name { font-size: 15px; }
}
@media (max-width: 400px) {
    .cm-nav { gap: 10px; }
    .cm-nav a { font-size: 11px; }
    .cm-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
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
    short = title[:22] + ("..." if len(title) > 22 else "")
    return (
        '<div class="cm-card-poster">'
        '<span class="material-icons-round" '
        'style="font-size:44px;color:#564d4d;">movie</span>'
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
# GENRE HELPERS
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
GENRE_LABELS = {"Science Fiction": "Sci-Fi"}

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
        "added": datetime.datetime.now().isoformat(),
    })
    return True


def remove_from_watchlist(movie_id):
    st.session_state.watchlist = [
        m for m in st.session_state.watchlist if m["id"] != movie_id
    ]


def is_in_watchlist(movie_id):
    return any(m["id"] == movie_id for m in st.session_state.watchlist)


# ============================================================
# TOP NAV
# ============================================================
current_page = st.session_state.page

def nav_class(p):
    return "active" if current_page == p else ""

wl_count = len(st.session_state.watchlist)
wl_label = f"My List ({wl_count})" if wl_count > 0 else "My List"

st.markdown(
    f"""
    <div class="cm-topbar">
        <div class="cm-brand">
            <span class="material-icons-round" style="font-size:28px;color:#E50914;">movie</span>
            <div class="cm-logo">CineMatch</div>
        </div>
        <div class="cm-nav">
            <a class="{nav_class('home')}" href="?page=home" target="_self">Home</a>
            <a class="{nav_class('browse')}" href="?page=browse" target="_self">Browse</a>
            <a class="{nav_class('search')}" href="?page=search" target="_self">Search</a>
            <a class="{nav_class('watchlist')}" href="?page=watchlist" target="_self">{wl_label}</a>
        </div>
        <div class="cm-icons">
            <span class="material-icons-round" style="font-size:22px;color:#B3B3B3;">notifications</span>
            <span class="material-icons-round" style="font-size:22px;color:#B3B3B3;">account_circle</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ENGINE
# ============================================================
df, sim = build_engine()


# ============================================================
# MOVIE GRID RENDERER
# ============================================================
def render_grid(results, query_title=None, show_similarity=True):
    """Render a list of movies as a responsive grid."""
    if results is None or len(results) == 0:
        return

    cards = []
    for _, row in results.iterrows():
        movie_id = row.get("id")
        rating = row.get("vote_average", None)
        rating_txt = (
            f"{rating:.1f}"
            if isinstance(rating, (int, float)) and rating else "N/A"
        )
        sim_val = row.get("similarity", None)
        sim_txt = f"{int(sim_val * 100)}% Similar" if sim_val is not None else ""

        # Poster
        poster = None
        if TMDB_API_KEY:
            poster = fetch_poster_by_id(movie_id)
            if not poster:
                poster = fetch_poster_by_title(row["title"])

        if poster:
            poster_html = f'<img class="cm-card-poster" src="{poster}" alt="poster">'
        else:
            poster_html = make_poster_placeholder(row["title"])

        in_wl = is_in_watchlist(movie_id) if movie_id is not None else False
        heart_color = "#E50914" if in_wl else "#B3B3B3"
        heart_icon = "favorite" if in_wl else "favorite_border"

        meta_html = (
            f'<span>{icon("star", 14)} {rating_txt}</span>'
            + (f'<span>{icon("bolt", 14)} {sim_txt}</span>' if sim_txt and show_similarity else "")
        )

        cards.append(
            f"""
            <div class="cm-card">
                {poster_html}
                <div class="cm-card-body">
                    <div class="cm-card-title">{row['title']}</div>
                    <div class="cm-card-meta">{meta_html}</div>
                    <div class="cm-card-overview">{str(row['overview'])}</div>
                </div>
            </div>
            """
        )

    # Render as one grid
    st.markdown(
        f'<div class="cm-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )

    # Action buttons row below each card — Streamlit-native
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    cols_per_row = min(5, max(2, len(results)))
    rows = [results.iloc[i:i + cols_per_row] for i in range(0, len(results), cols_per_row)]
    for row_df in rows:
        cols = st.columns(cols_per_row)
        for i, (_, row) in enumerate(row_df.iterrows()):
            with cols[i]:
                movie_id = row.get("id")
                if movie_id is None:
                    continue
                in_wl = is_in_watchlist(movie_id)

                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button(
                        "♥" if in_wl else "♡",
                        key=f"heart_{movie_id}_{row['title']}",
                        use_container_width=True,
                    ):
                        if in_wl:
                            remove_from_watchlist(movie_id)
                        else:
                            poster = fetch_poster_by_id(movie_id) if TMDB_API_KEY else None
                            add_to_watchlist(
                                movie_id, row["title"], poster,
                                f"{row.get('vote_average', 0):.1f}",
                            )
                        st.rerun()
                with c2:
                    if st.button(
                        "More",
                        key=f"more_{movie_id}_{row['title']}",
                        use_container_width=True,
                    ):
                        st.session_state.last_query = row["title"]
                        st.session_state.page = "search"
                        st.query_params["page"] = "search"
                        st.rerun()


# ============================================================
# PAGE: HOME
# ============================================================
def render_home():
    st.markdown(
        f"""
        <div class="cm-search-hero">
            <div class="cm-hero-title">{icon('local_movies', 28, '#FFFFFF')} Find Your Next Obsession</div>
            <div class="cm-hero-sub">Browse by genre, discover today's pick, or search for your next favorite.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Movie of the Day
    st.markdown('<div class="cm-section">Movie of the Day</div>', unsafe_allow_html=True)
    st.markdown('<div class="cm-section-underline"></div>', unsafe_allow_html=True)

    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top_rated = df[df["vote_average"] >= 7.5] if "vote_average" in df.columns else df
    if not top_rated.empty:
        motd = top_rated.sample(1).to_frame().T.reset_index(drop=True)
        motd["similarity"] = None
        render_grid(motd, show_similarity=False)

    # Trending
    if TMDB_API_KEY:
        st.markdown('<div class="cm-section">Trending This Week</div>', unsafe_allow_html=True)
        st.markdown('<div class="cm-section-underline"></div>', unsafe_allow_html=True)

        trending = fetch_trending_movies(8)
        if trending:
            cards = []
            for m in trending:
                poster = m["poster"] or ""
                rating = m["rating"] or 0
                release = (m["release"] or "")[:4]
                poster_tag = (
                    f'<img class="cm-card-poster" src="{poster}" alt="poster">'
                    if poster
                    else f'<div class="cm-card-poster">{icon("movie", 40, "#564d4d")}</div>'
                )
                cards.append(
                    f"""
                    <div class="cm-card">
                        {poster_tag}
                        <div class="cm-card-body">
                            <div class="cm-card-title">{m['title']}</div>
                            <div class="cm-card-meta">
                                <span>{icon('star', 14)} {rating:.1f}</span>
                                <span>{release}</span>
                            </div>
                        </div>
                    </div>
                    """
                )
            st.markdown(
                f'<div class="cm-grid">{"".join(cards)}</div>',
                unsafe_allow_html=True,
            )

    # Quick browse
    st.markdown('<div class="cm-section">Explore by Genre</div>', unsafe_allow_html=True)
    st.markdown('<div class="cm-section-underline"></div>', unsafe_allow_html=True)
    st.markdown(
        '<a href="?page=browse" target="_self" '
        'style="display:inline-block;background:#E50914;color:#fff;'
        'padding:12px 26px;border-radius:4px;text-decoration:none;'
        'font-weight:700;letter-spacing:0.5px;font-size:14px;">BROWSE GENRES</a>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: BROWSE
# ============================================================
def render_browse():
    if st.session_state.selected_genre:
        genre = st.session_state.selected_genre
        st.markdown(
            '<a href="?page=browse&g=none" target="_self" '
            'style="display:inline-block;background:#E50914;color:#fff;'
            'padding:10px 20px;border-radius:4px;text-decoration:none;'
            'font-weight:700;font-size:13px;">← BACK TO ALL GENRES</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cm-section">{GENRE_LABELS.get(genre, genre)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="cm-section-underline"></div>', unsafe_allow_html=True)

        movies = get_genre_movies(df, genre, n=20)
        if movies.empty:
            st.info(f"No movies found in {genre}.")
        else:
            st.caption(f"Top {len(movies)} rated movies in {genre}")
            render_grid(movies, show_similarity=False)
        return

    # Hero
    st.markdown(
        f"""
        <div class="cm-search-hero">
            <div class="cm-hero-title">{icon('explore', 28, '#FFFFFF')} Browse by Genre</div>
            <div class="cm-hero-sub">Don't know what to watch? Pick a genre and we'll show you the best.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Genre tiles grid
    tiles = []
    for genre in BROWSE_GENRES:
        count = count_genre_movies(df, genre)
        label = GENRE_LABELS.get(genre, genre)
        icon_name = GENRE_ICONS.get(genre, "movie")
        tiles.append(
            f"""
            <a href="?page=browse&genre={genre}" target="_self"
               style="text-decoration:none;">
                <div class="cm-genre-tile">
                    <div class="cm-genre-icon">{icon(icon_name, 30, '#FFFFFF')}</div>
                    <div class="cm-genre-name">{label}</div>
                    <div class="cm-genre-count">{count:,} movies</div>
                </div>
            </a>
            """
        )
    st.markdown(
        f'<div class="cm-genre-grid">{"".join(tiles)}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: SEARCH
# ============================================================
def render_search():
    st.markdown(
        f"""
        <div class="cm-search-hero">
            <div class="cm-hero-title">{icon('search', 28, '#FFFFFF')} Search Movies</div>
            <div class="cm-hero-sub">Type a movie you love — we'll match you with your next favorite.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        hist_html = " &nbsp;•&nbsp; ".join(st.session_state.history)
        st.markdown(
            f'<div class="cm-history">{icon("history", 16, "#B3B3B3")} Recent: {hist_html}</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder="Try: The Dark Knight, Avatar, Inception...",
            label_visibility="collapsed",
            key="search_query",
            value=st.session_state.last_query or "",
        )
    with col2:
        n_results = st.selectbox(
            "Show", [5, 10, 15, 20], index=1,
            label_visibility="collapsed", key="search_n",
        )

    st.markdown(
        f'<div style="color:#B3B3B3;font-size:13px;margin:14px 0 6px 0;">'
        f'{icon("tune", 16, "#B3B3B3")} Filter by genre (optional)</div>',
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
                f' {icon("lightbulb", 16, "#B3B3B3")} Suggestions: '
                + " &nbsp;•&nbsp; ".join(suggestions)
            )
            st.markdown(
                f'<div style="color:#B3B3B3;font-size:13px;margin-top:-8px;">'
                f'{sugg_html}</div>',
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
            st.session_state.last_query = clean_q

            with st.spinner("Finding your matches..."):
                results = recommend(
                    df, sim, query, n=n_results,
                    genre_filter=selected_genres,
                )

            if results.empty:
                st.error(f'No movies found matching "{query}". Try another title.')
            else:
                st.session_state.last_results = results.to_dict("records")
                st.markdown(
                    f'<div class="cm-section">Because you liked <em>{clean_q}</em></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="cm-section-underline"></div>', unsafe_allow_html=True)
                render_grid(results)


# ============================================================
# PAGE: WATCHLIST
# ============================================================
def render_watchlist():
    wl = st.session_state.watchlist

    st.markdown(
        f'<div class="cm-section">{icon("favorite", 26)} My List</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cm-section-underline"></div>', unsafe_allow_html=True)

    if not wl:
        st.markdown(
            f"""
            <div class="cm-empty">
                {icon('movie_filter', 48, '#564d4d')}
                <p style="margin-top:14px;font-size:15px;">
                    Your list is empty. Search for movies and tap the heart to save them here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(wl)} titles · sorted by added (newest)")

    # Sort control
    sort_by = st.selectbox(
        "Sort by",
        ["Added (Newest)", "Rating (High to Low)", "Title (A-Z)"],
        index=0, label_visibility="collapsed",
    )
    if sort_by == "Added (Newest)":
        wl_sorted = sorted(wl, key=lambda m: m.get("added", ""), reverse=True)
    elif sort_by == "Rating (High to Low)":
        wl_sorted = sorted(wl, key=lambda m: m.get("rating") or "0", reverse=True)
    else:
        wl_sorted = sorted(wl, key=lambda m: m.get("title", ""))

    # Cards grid
    cards = []
    for m in wl_sorted:
        poster = m.get("poster")
        poster_html = (
            f'<img class="cm-card-poster" src="{poster}" alt="poster">'
            if poster
            else make_poster_placeholder(m["title"])
        )
        cards.append(
            f"""
            <div class="cm-card">
                <div style="position:relative;">
                    {poster_html}
                    <a href="?page=watchlist&remove={m['id']}" target="_self"
                       style="position:absolute;top:10px;right:10px;
                              background:rgba(229,9,20,0.95);color:#fff;
                              border-radius:50%;width:30px;height:30px;
                              display:flex;align-items:center;justify-content:center;
                              text-decoration:none;font-weight:900;font-size:16px;">
                        ×
                    </a>
                    <div style="position:absolute;top:10px;left:10px;
                                background:rgba(0,0,0,0.85);color:#fff;
                                border-radius:4px;padding:3px 8px;
                                font-size:12px;font-weight:800;">
                        {m.get('rating') or 'N/A'}
                    </div>
                </div>
                <div class="cm-card-body">
                    <div class="cm-card-title">{m['title']}</div>
                    <div class="cm-card-meta"></div>
                </div>
            </div>
            """
        )
    st.markdown(
        f'<div class="cm-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )

    # Clear all
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    if st.button("CLEAR MY LIST", key="clear_list"):
        st.session_state.watchlist = []
        st.rerun()


# ============================================================
# HANDLE QUERY PARAMS (genre + remove)
# ============================================================
# Reset genre from URL
if st.query_params.get("g") == "none":
    st.session_state.selected_genre = None

# Set genre from URL
if "genre" in st.query_params and st.query_params["genre"]:
    st.session_state.selected_genre = st.query_params["genre"]

# Remove from watchlist from URL
if "remove" in st.query_params and st.query_params["remove"]:
    try:
        rid = int(st.query_params["remove"])
        remove_from_watchlist(rid)
        # Clean the URL
        st.query_params["page"] = "watchlist"
        if "remove" in st.query_params:
            del st.query_params["remove"]
    except Exception:
        pass


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
    <div class="cm-footer">
        CINEMATCH &nbsp;•&nbsp; Powered by NLP & TF-IDF &nbsp;•&nbsp;
        Built with {icon('favorite', 14)} at TekHer AI Bootcamp
    </div>
    """,
    unsafe_allow_html=True,
)