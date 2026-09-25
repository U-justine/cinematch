"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
Pages: Home, Browse, Search, Watchlist, Collections, Profile.
Built to match the reference mockups pixel-for-pixel:
  - Sidebar nav              -> "My Watchlist" mockup
  - Genre grid (Browse)      -> "Browse by Genre" mockup
  - Search results grid      -> "Because you liked..." mockup
CSS lives in style.css (loaded, not inlined) per request.
"""

import ast
import datetime
import os
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
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# LOAD EXTERNAL STYLESHEET
# ============================================================
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(
        '<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">'
        '<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


# ============================================================
# ICON HELPERS (Material Icons only — never emoji)
# ============================================================
def icon(name: str, size: int = 20, color: str = "currentColor", outlined: bool = False) -> str:
    cls = "material-icons-outlined" if outlined else "material-icons-round"
    return (
        f'<span class="{cls} mi" style="font-size:{size}px;color:{color};">{name}</span>'
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
        return None, None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        poster = f"{TMDB_IMG_BASE}{data['poster_path']}" if data.get("poster_path") else None
        backdrop = f"{TMDB_BACKDROP_BASE}{data['backdrop_path']}" if data.get("backdrop_path") else None
        return poster, backdrop
    except Exception:
        return None, None


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


GENRE_ID_MAP = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
    "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14,
    "History": 36, "Horror": 27, "Music": 10402, "Mystery": 9648,
    "Romance": 10749, "Science Fiction": 878, "Thriller": 53,
    "War": 10752, "Western": 37,
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_genre_backdrop(genre_name: str):
    """Representative backdrop image for a genre, used on the Browse cards."""
    if not TMDB_API_KEY:
        return None
    gid = GENRE_ID_MAP.get(genre_name)
    if not gid:
        return None
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {"api_key": TMDB_API_KEY, "with_genres": gid, "sort_by": "popularity.desc"}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("backdrop_path"):
            return f"{TMDB_BACKDROP_BASE}{results[0]['backdrop_path']}"
    except Exception:
        pass
    return None


def poster_placeholder_style(seed_text: str) -> str:
    """Deterministic dark gradient fallback so cards never look broken without an API key."""
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
# GENRE CONFIG (icons + order match the "Browse by Genre" mockup)
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
    "Animation": "movie_filter",
    "Adventure": "explore",
    "Crime": "shield",
    "Documentary": "camera_alt",
}

GENRE_LABELS = {"Science Fiction": "Sci-Fi"}

BROWSE_GENRES = list(GENRE_ICONS.keys())


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
# SIDEBAR NAVIGATION — matches "My Watchlist" mockup exactly
# ============================================================
NAV_ITEMS = [
    ("home", "Home", "home"),
    ("browse", "Browse", "search"),
    ("watchlist", "My Watchlist", "favorite"),
    ("collections", "Collections", "video_library"),
    ("profile", "Profile", "person"),
]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">
                <span class="material-icons-round" style="color:white;font-size:19px;">theaters</span>
            </div>
            <div class="sidebar-logo-text"><span class="cine">Cine</span><span class="match">Match</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for key, label, icon_name in NAV_ITEMS:
        is_active = st.session_state.page == key
        cols = st.columns([1, 6])
        with cols[0]:
            st.markdown(
                f'<div style="padding-top:9px;">{icon(icon_name, 19, "var(--red)" if is_active else "#9a9a9a")}</div>',
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
        st.markdown(f'<div style="padding-top:9px;">{icon("search", 19, "#9a9a9a")}</div>', unsafe_allow_html=True)
    with scols[1]:
        if st.button("Search", key="nav_search_link", use_container_width=True,
                     type="primary" if st.session_state.page == "search" else "secondary"):
            st.session_state.page = "search"
            st.rerun()

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    lcols = st.columns([1, 6])
    with lcols[0]:
        st.markdown(f'<div style="padding-top:9px;">{icon("logout", 19, "#9a9a9a")}</div>', unsafe_allow_html=True)
    with lcols[1]:
        st.button("Log Out", key="nav_logout", use_container_width=True)


# ============================================================
# LOAD ENGINE
# ============================================================
df, sim = build_engine()


# ============================================================
# SHARED RENDER HELPERS
# ============================================================
def render_shelf_card(title, poster, rating, year=None):
    poster_html = (
        f'<img class="shelf-poster" src="{poster}" alt="{title}">'
        if poster else
        f'<div class="shelf-poster" style="background:{poster_placeholder_style(title)};'
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
        poster, _ = fetch_poster_by_id(movie_id)
        poster = poster or fetch_poster_by_title(row["title"])

    poster_html = (
        f'<img class="result-poster" src="{poster}" alt="{row["title"]}">'
        if poster else
        f'<div class="result-poster" style="background:{poster_placeholder_style(row["title"])};'
        f'display:flex;align-items:center;justify-content:center;">{icon("movie", 34, "rgba(255,255,255,0.5)")}</div>'
    )

    match_html = ""
    if show_similarity and sim_val is not None:
        pct = int(sim_val * 100)
        match_html = f'<span class="match">{pct}% Similar</span>'

    in_wl = is_in_watchlist(movie_id)
    heart_color = "var(--red)" if in_wl else "#888"

    st.markdown(
        f"""
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
                <div class="result-heart {'active' if in_wl else ''}">{icon('favorite', 15, heart_color)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if movie_id is not None:
        label = "♥ In Watchlist — tap to remove" if in_wl else "Add to Watchlist"
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
        """
        <div class="home-hero">
            <div class="home-hero-label">Welcome to CineMatch</div>
            <h1 class="home-hero-title">Find Your Next Obsession</h1>
            <p class="home-hero-sub">AI-powered recommendations. Endless stories. Discover movies that match your mood.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="section-title">{icon("today", 20, "var(--red)")} Movie of the Day</div>', unsafe_allow_html=True)

    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top_rated = df[df["vote_average"] >= 7.5] if "vote_average" in df.columns else df
    if not top_rated.empty:
        motd = top_rated.sample(1).iloc[0]
        poster = None
        if TMDB_API_KEY:
            poster, _ = fetch_poster_by_id(motd.get("id"))
            poster = poster or fetch_poster_by_title(motd["title"])
        poster_html = (
            f'<img class="motd-poster" src="{poster}" alt="poster">'
            if poster else
            f'<div class="motd-poster" style="background:{poster_placeholder_style(motd["title"])};'
            f'display:flex;align-items:center;justify-content:center;">{icon("movie", 32, "rgba(255,255,255,0.5)")}</div>'
        )
        rating = motd.get("vote_average", 0)
        st.markdown(
            f"""
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
            """,
            unsafe_allow_html=True,
        )

    if TMDB_API_KEY:
        st.markdown(f'<div class="section-title">{icon("local_fire_department", 20, "var(--red)")} Trending This Week</div>', unsafe_allow_html=True)
        trending = fetch_trending_movies(12)
        if trending:
            cards = "".join([
                render_shelf_card(m["title"], m["poster"], m["rating"] or 0, m.get("release"))
                for m in trending
            ])
            st.markdown(f'<div class="shelf">{cards}</div>', unsafe_allow_html=True)
        else:
            st.info("Trending unavailable right now.")

    st.markdown(f'<div class="section-title">{icon("explore", 20, "var(--red)")} Or Explore by Genre</div>', unsafe_allow_html=True)
    if st.button("BROWSE ALL GENRES →", use_container_width=True, key="home_browse", type="primary"):
        st.session_state.page = "browse"
        st.rerun()


# ============================================================
# PAGE: BROWSE — matches "Browse by Genre" mockup exactly
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
                icon_name = GENRE_ICONS.get(genre, "movie")

                backdrop = fetch_genre_backdrop(genre) if TMDB_API_KEY else None
                bg_style = (
                    f"background-image: url('{backdrop}');"
                    if backdrop else
                    f"background-image: {poster_placeholder_style(genre)};"
                )

                st.markdown(
                    f"""
                    <div class="genre-card" style="{bg_style}">
                        <div class="genre-scrim">
                            <div class="genre-icon">{icon(icon_name, 30, "white", outlined=True)}</div>
                            <div>
                                <div class="genre-name">{label}</div>
                                <div class="genre-count"><span class="dot"></span>{count:,} movies</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {label}", key=f"genre_{genre}", use_container_width=True):
                    st.session_state.selected_genre = genre
                    st.rerun()


# ============================================================
# PAGE: SEARCH — matches "Because you liked..." mockup exactly
# ============================================================
def render_search():
    with st.container():
        col_icon, col_input, col_btn = st.columns([0.4, 8, 0.8])
        with col_icon:
            st.markdown(f'<div style="padding-top:12px;">{icon("search", 20, "#888")}</div>', unsafe_allow_html=True)
        with col_input:
            query = st.text_input(
                "Search", placeholder="Try: The Dark Knight, Inception, Avatar…",
                label_visibility="collapsed", key="search_query",
            )
        with col_btn:
            go = st.button("🔍", key="search_go_btn", use_container_width=True, type="primary")

    if st.session_state.history:
        hist = "  ·  ".join(st.session_state.history)
        st.markdown(
            f'<div style="color:#888;font-size:12.5px;margin:-14px 0 18px 2px;">'
            f'{icon("history", 14, "#888")} Recent: {hist}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Filter by genre (optional)"):
        selected_genres = st.multiselect(
            "Genres", options=BROWSE_GENRES, default=st.session_state.genre_filter,
            label_visibility="collapsed", format_func=lambda g: GENRE_LABELS.get(g, g),
        )
        st.session_state.genre_filter = selected_genres

    if query and len(query) > 1:
        suggestions = search_titles(df, query)
        if suggestions:
            st.markdown(
                f'<div style="color:#888;font-size:12.5px;margin:2px 0 18px 2px;">'
                f'{icon("lightbulb", 14, "#888")} Suggestions: {"  ·  ".join(suggestions)}</div>',
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
                    f'<div class="results-heading">Because you liked <em>"{query.title()}"</em></div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(4)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 4]:
                        render_result_card(row, show_similarity=True)


# ============================================================
# PAGE: WATCHLIST — matches "My Watchlist" mockup exactly
# ============================================================
def render_watchlist():
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-title-row">
                <h1 class="page-title">My Watchlist</h1>
                {icon('favorite', 26, 'var(--red)')}
            </div>
        </div>
        """,
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
            f"""
            <div class="empty-state">
                {icon('movie_filter', 44, '#444')}
                <p style="margin-top:16px;font-size:15px;">
                    Your watchlist is empty.<br>
                    Search for movies and tap <strong>Add to Watchlist</strong> to save them here.
                </p>
            </div>
            """,
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
                f'<div class="watch-poster" style="background:{poster_placeholder_style(m["title"])};'
                f'display:flex;align-items:center;justify-content:center;">{icon("movie", 28, "rgba(255,255,255,0.5)")}</div>'
            )
            genres_txt = ", ".join((m.get("genres") or [])[:2])
            year_genres = " · ".join([x for x in [m.get("year"), genres_txt] if x])

            st.markdown(
                f"""
                <div class="watch-card">
                    <div class="watch-poster-wrap">
                        {poster_html}
                        <div class="watch-rating-badge">{icon('star', 11, 'var(--gold)')} {m.get('rating') or 'N/A'}</div>
                        <div class="watch-remove-badge">{icon('close', 14, 'white')}</div>
                    </div>
                    <div class="watch-title">{m['title']}</div>
                    <div class="watch-year">{year_genres}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("Remove", key=f"rm_{m['id']}", use_container_width=True):
                    remove_from_watchlist(m["id"])
                    st.rerun()
            with bc2:
                if st.button("More like this", key=f"more_{m['id']}", use_container_width=True):
                    st.session_state.page = "search"
                    st.session_state["search_query"] = m["title"]
                    st.rerun()


# ============================================================
# PAGE: COLLECTIONS (placeholder — sidebar link target)
# ============================================================
def render_collections():
    st.markdown('<h1 class="page-title">Collections</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Curated groups of titles. Coming soon.</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="empty-state">
            {icon('video_library', 44, '#444')}
            <p style="margin-top:16px;font-size:15px;">Collections aren't set up yet — check back soon.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: PROFILE (placeholder — sidebar link target)
# ============================================================
def render_profile():
    st.markdown('<h1 class="page-title">Profile</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Your account details.</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="empty-state">
            {icon('person', 44, '#444')}
            <p style="margin-top:16px;font-size:15px;">Profile settings aren't set up yet.</p>
        </div>
        """,
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
    f"""
    <div style="text-align:center;padding:40px 0 10px;color:#555;font-size:12px;">
        CINEMATCH · Powered by NLP & TF-IDF · Built with {icon('favorite', 12, '#555')} at TekHer AI Bootcamp
    </div>
    """,
    unsafe_allow_html=True,
)
