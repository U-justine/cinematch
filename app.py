"""
CineMatch — Netflix-style movie recommender
Polished version: Material Icons only, sidebar navigation, improved caching.
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
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "page": "home",
    "watchlist": [],
    "history": [],
    "genre_filter": [],
    "selected_genre": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CSS
# ============================================================
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
:root {
    --red: #E50914;
    --bg: #0f0f0f;
    --card: #1a1a1a;
    --muted: #a0a0a0;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: #fff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #141414 !important;
    border-right: 1px solid #222;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #ccc !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(229,9,20,0.15) !important;
    color: #fff !important;
}

/* Logo */
.logo {
    font-size: 24px;
    font-weight: 900;
    color: var(--red);
    letter-spacing: -0.5px;
    margin-bottom: 28px;
}

/* Section titles */
.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #fff;
    margin: 12px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-sub {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 20px;
}

/* Genre cards */
.genre-card {
    background: linear-gradient(145deg, #1c1c1c, #141414);
    border-radius: 10px;
    border-left: 4px solid var(--red);
    height: 120px;
    position: relative;
    transition: all 0.22s ease;
    margin-bottom: 6px;
}
.genre-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(229,9,20,0.22);
}
.genre-inner {
    padding: 16px 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.genre-name {
    font-size: 18px;
    font-weight: 800;
    color: #fff;
    margin: 0;
}
.genre-count {
    font-size: 12px;
    color: var(--muted);
    margin-top: 3px;
}
.genre-count span { color: var(--red); font-weight: 700; }
.genre-icon {
    position: absolute;
    top: 14px;
    right: 16px;
    font-size: 28px !important;
    color: rgba(255,255,255,0.15);
}

/* Result cards (Search / Genre detail) */
.result-card {
    background: #1a1a1a;
    border-radius: 10px;
    display: flex;
    overflow: hidden;
    border: 1px solid #2a2a2a;
    margin-bottom: 14px;
    min-height: 150px;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: var(--red); }
.result-poster {
    width: 100px;
    min-width: 100px;
    object-fit: cover;
    background: #111;
}
.result-body {
    padding: 12px 14px;
    flex: 1;
    display: flex;
    flex-direction: column;
}
.result-title {
    font-size: 15px;
    font-weight: 800;
    color: #fff;
    margin: 0 0 2px 0;
}
.result-year {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 5px;
}
.result-meta {
    font-size: 13px;
    color: #f5c518;
    font-weight: 600;
    margin-bottom: 6px;
}
.result-meta .sim {
    color: var(--red);
    margin-left: 8px;
}
.result-overview {
    font-size: 12.5px;
    color: #b0b0b0;
    line-height: 1.4;
    flex: 1;
}

/* Inputs */
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #fff !important;
    border-radius: 8px !important;
    padding: 11px 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.25) !important;
}

.stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
}
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background-color: var(--red) !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def icon(name: str, size: int = 20, color: str = "#E50914") -> str:
    return (
        f'<span class="material-icons-round" '
        f'style="font-size:{size}px;color:{color};vertical-align:middle;">'
        f'{name}</span>'
    )

def placeholder(title: str, height: str = "150px") -> str:
    short = (title[:18] + "…") if len(title) > 18 else title
    return (
        f'<div style="width:100%;height:{height};'
        f'background:linear-gradient(135deg,#E50914,#5c0000);'
        f'display:flex;align-items:center;justify-content:center;'
        f'color:white;font-size:11px;font-weight:700;text-align:center;padding:6px;">'
        f'{short}</div>'
    )


# ============================================================
# TMDb + DATA
# ============================================================
def get_tmdb_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None

TMDB_API_KEY = get_tmdb_key()
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending(limit: int = 10):
    if not TMDB_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/trending/movie/week",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )
        r.raise_for_status()
        return [
            {
                "id": m["id"],
                "title": m["title"],
                "poster": f"{TMDB_IMG}{m['poster_path']}" if m.get("poster_path") else None,
                "rating": m.get("vote_average", 0),
                "year": (m.get("release_date") or "")[:4],
            }
            for m in r.json().get("results", [])[:limit]
        ]
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(movie_id=None, title=None):
    """Single cached lookup – avoids repeated API hits."""
    if not TMDB_API_KEY:
        return None
    try:
        if movie_id:
            r = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": TMDB_API_KEY},
                timeout=8,
            )
            r.raise_for_status()
            path = r.json().get("poster_path")
            if path:
                return f"{TMDB_IMG}{path}"
        if title:
            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_API_KEY, "query": title},
                timeout=8,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"{TMDB_IMG}{results[0]['poster_path']}"
    except Exception:
        pass
    return None

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "IgorNascAlves/data-science-primeiros-passos/"
    "master/tmdb_5000_movies.csv"
)

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv(DATASET_URL)
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

def _parse(text):
    try:
        return " ".join(i["name"] for i in ast.literal_eval(text)).lower()
    except Exception:
        return ""

@st.cache_resource(show_spinner="Loading CineMatch engine…")
def build_engine():
    df = load_data().copy().dropna(subset=["title", "overview"])
    df["genres"] = df["genres"].apply(_parse)
    df["keywords"] = df["keywords"].apply(_parse)
    df["overview"] = df["overview"].str.lower().str.strip()
    df["soup"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]
    keep = [c for c in ["id", "title", "overview", "genres", "vote_average", "release_date", "soup"] if c in df.columns]
    df = df[keep].reset_index(drop=True)
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=2)
    matrix = tfidf.fit_transform(df["soup"])
    return df, cosine_similarity(matrix, matrix)

def recommend(df, sim, title, n=8, genre_filter=None):
    title = title.lower().strip()
    matches = df[df["title"].str.lower() == title]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1 : n * 4 + 1]
    result = df.iloc[[i for i, _ in scores]][
        ["id", "title", "overview", "genres", "vote_average", "release_date"]
    ].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    if genre_filter:
        mask = result["genres"].apply(lambda g: any(x.lower() in g for x in genre_filter))
        result = result[mask]
    return result.head(n)


# ============================================================
# GENRE + WATCHLIST HELPERS
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
BROWSE_GENRES = list(GENRE_ICONS.keys())

def get_genre_movies(df, genre, n=15):
    mask = df["genres"].str.contains(genre.lower(), na=False)
    return df[mask].sort_values("vote_average", ascending=False).head(n)

def count_genre(df, genre):
    return int(df["genres"].str.contains(genre.lower(), na=False).sum())

def add_to_watchlist(mid, title, poster=None, rating=None, year=None):
    if any(m["id"] == mid for m in st.session_state.watchlist):
        return
    st.session_state.watchlist.append({
        "id": mid, "title": title, "poster": poster,
        "rating": rating, "year": year,
    })

def remove_from_watchlist(mid):
    st.session_state.watchlist = [m for m in st.session_state.watchlist if m["id"] != mid]

def is_in_watchlist(mid):
    return any(m["id"] == mid for m in st.session_state.watchlist)


# ============================================================
# LOAD ENGINE
# ============================================================
df, sim = build_engine()


# ============================================================
# SIDEBAR NAVIGATION (Material Icons only)
# ============================================================
with st.sidebar:
    st.markdown('<div class="logo">CINEMATCH</div>', unsafe_allow_html=True)

    if st.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.page = "home"
        st.session_state.selected_genre = None
        st.rerun()

    if st.button("Browse", use_container_width=True, key="nav_browse"):
        st.session_state.page = "browse"
        st.session_state.selected_genre = None
        st.rerun()

    if st.button("Search", use_container_width=True, key="nav_search"):
        st.session_state.page = "search"
        st.rerun()

    if st.button(f"Watchlist ({len(st.session_state.watchlist)})", use_container_width=True, key="nav_wl"):
        st.session_state.page = "watchlist"
        st.rerun()

    st.markdown("---")
    st.caption(f"{len(st.session_state.watchlist)} titles saved")


# ============================================================
# RENDER HELPERS
# ============================================================
def render_result_card(row, show_sim=True):
    mid = row.get("id")
    title = row["title"]
    year = str(row.get("release_date", ""))[:4] if pd.notna(row.get("release_date")) else ""
    rating = row.get("vote_average") or 0
    overview = str(row.get("overview", ""))[:130] + "…"
    sim_pct = int(row.get("similarity", 0) * 100) if show_sim and "similarity" in row else None

    poster = fetch_poster(mid, title)
    poster_html = (
        f'<img class="result-poster" src="{poster}" alt="{title}">'
        if poster else placeholder(title, "150px")
    )

    meta = f'{icon("star", 14, "#f5c518")} {rating:.1f}'
    if sim_pct is not None:
        meta += f' <span class="sim">{sim_pct}% Similar</span>'

    st.markdown(f"""
    <div class="result-card">
        {poster_html}
        <div class="result-body">
            <div class="result-title">{title}</div>
            <div class="result-year">{year}</div>
            <div class="result-meta">{meta}</div>
            <div class="result-overview">{overview}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Watchlist toggle
    in_list = is_in_watchlist(mid)
    btn_label = "Remove from Watchlist" if in_list else "Add to Watchlist"
    if st.button(btn_label, key=f"wl_{mid}_{title[:15]}", use_container_width=False):
        if in_list:
            remove_from_watchlist(mid)
        else:
            add_to_watchlist(mid, title, poster, f"{rating:.1f}", year)
        st.rerun()


# ============================================================
# PAGES
# ============================================================
def render_home():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a0505,#0f0f0f);
                border-radius:12px;padding:36px 28px;margin-bottom:24px;">
        <div style="color:var(--red);font-size:12px;font-weight:700;
                    letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;">
            Welcome to CineMatch
        </div>
        <div style="font-size:32px;font-weight:900;color:#fff;margin-bottom:8px;">
            Find Your Next Obsession
        </div>
        <div style="color:#ccc;font-size:14px;max-width:400px;">
            AI-powered recommendations. Discover movies that match your mood.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Movie of the Day
    st.markdown(f'<div class="section-title">{icon("today", 22)} Movie of the Day</div>', unsafe_allow_html=True)
    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top = df[df["vote_average"] >= 7.5]
    if not top.empty:
        m = top.sample(1).iloc[0]
        poster = fetch_poster(m.get("id"), m["title"])
        rating = m.get("vote_average", 0)
        c1, c2 = st.columns([1, 3])
        with c1:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(placeholder(m["title"], "220px"), unsafe_allow_html=True)
        with c2:
            st.markdown(f"### {m['title']}")
            st.markdown(f'{icon("star", 16, "#f5c518")} **{rating:.1f}/10**', unsafe_allow_html=True)
            st.write(str(m["overview"])[:300] + "…")

    # Trending
    if TMDB_API_KEY:
        st.markdown(
            f'<div class="section-title" style="margin-top:28px;">{icon("local_fire_department", 22)} Trending This Week</div>',
            unsafe_allow_html=True,
        )
        trending = fetch_trending(10)
        if trending:
            cols = st.columns(5)
            for i, t in enumerate(trending):
                with cols[i % 5]:
                    if t["poster"]:
                        st.image(t["poster"], use_container_width=True)
                    st.caption(f"**{t['title'][:20]}**  \n{icon('star', 12, '#f5c518')} {t['rating']:.1f}", unsafe_allow_html=True)


def render_browse():
    if st.session_state.selected_genre:
        genre = st.session_state.selected_genre
        if st.button("Back to all genres"):
            st.session_state.selected_genre = None
            st.rerun()
        label = GENRE_LABELS.get(genre, genre)
        st.markdown(
            f'<div class="section-title">{icon(GENRE_ICONS.get(genre, "movie"), 22)} {label}</div>',
            unsafe_allow_html=True,
        )
        movies = get_genre_movies(df, genre, 12)
        for _, row in movies.iterrows():
            render_result_card(row, show_sim=False)
        return

    st.markdown(f'<div class="section-title">{icon("explore", 22)} Browse by Genre</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore movies by genre. Discover stories that match your mood.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, genre in enumerate(BROWSE_GENRES):
        with cols[i % 4]:
            count = count_genre(df, genre)
            label = GENRE_LABELS.get(genre, genre)
            icon_name = GENRE_ICONS.get(genre, "movie")
            st.markdown(f"""
            <div class="genre-card">
                <span class="material-icons-round genre-icon">{icon_name}</span>
                <div class="genre-inner">
                    <div class="genre-name">{label}</div>
                    <div class="genre-count"><span>●</span> {count:,} movies</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {label}", key=f"genre_{genre}", use_container_width=True):
                st.session_state.selected_genre = genre
                st.rerun()


def render_search():
    st.markdown(f'<div class="section-title">{icon("search", 22)} Search Movies</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Search",
        placeholder="Try: The Dark Knight, Inception, Avatar…",
        label_visibility="collapsed",
    )
    n = st.selectbox("Number of results", [6, 8, 12], index=1, label_visibility="collapsed")

    if st.button("Find My Match", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a movie title.")
            return
        with st.spinner("Finding your matches…"):
            results = recommend(df, sim, query, n=n)
        if results.empty:
            st.error(f'No movies found matching “{query}”.')
            return
        st.markdown(
            f'<div class="section-title" style="margin-top:18px;">'
            f'{icon("auto_awesome", 22)} Because you liked “{query.title()}”</div>',
            unsafe_allow_html=True,
        )
        for _, row in results.iterrows():
            render_result_card(row, show_sim=True)


def render_watchlist():
    st.markdown(
        f'<div class="section-title">{icon("favorite", 22)} My Watchlist</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-sub">{len(st.session_state.watchlist)} titles saved</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.watchlist:
        st.markdown(f"""
        <div style="text-align:center;padding:50px 20px;background:#1a1a1a;
                    border-radius:12px;color:#888;">
            {icon("movie_filter", 48, "#555")}
            <p style="margin-top:14px;font-size:14px;">
                Your watchlist is empty.<br>
                Search for movies and add them here.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    cols = st.columns(5)
    for i, m in enumerate(st.session_state.watchlist):
        with cols[i % 5]:
            if m.get("poster"):
                st.image(m["poster"], use_container_width=True)
            else:
                st.markdown(placeholder(m["title"], "180px"), unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:6px;">
                <div style="font-size:13px;font-weight:700;color:#fff;line-height:1.3;">{m["title"]}</div>
                <div style="font-size:11px;color:#a0a0a0;margin-top:2px;">
                    {m.get("year", "")} · {icon("star", 11, "#f5c518")} {m.get("rating", "N/A")}
                </div>
            </div>
            """, unsafe_allow_html=True)
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

st.markdown(
    f'<div style="text-align:center;color:#555;font-size:12px;margin-top:48px;">'
    f'CINEMATCH · Powered by NLP & TF-IDF · {icon("favorite", 12)}</div>',
    unsafe_allow_html=True,
)