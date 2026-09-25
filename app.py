"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
"""

import ast
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
# NETFLIX-STYLE CSS + MATERIAL ICONS
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

/* ---------- Global ---------- */
.stApp {
    background-color: var(--netflix-black);
    color: var(--netflix-white);
    font-family: 'Helvetica Neue', Arial, sans-serif;
}

/* ---------- Material Icons ---------- */
.material-icons-round {
    font-family: 'Material Icons Round';
    font-weight: normal;
    font-style: normal;
    font-size: 22px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    vertical-align: middle;
    color: var(--netflix-red);
}

/* ---------- Header ---------- */
.cinematch-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    background: linear-gradient(180deg, rgba(0,0,0,0.95) 0%, rgba(20,20,20,0) 100%);
    position: sticky;
    top: 0;
    z-index: 100;
}

.cinematch-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.cinematch-logo {
    font-size: 36px;
    font-weight: 900;
    color: var(--netflix-red);
    letter-spacing: -1px;
    text-transform: uppercase;
    text-shadow: 0 2px 10px rgba(0,0,0,0.8);
    margin: 0;
}

.cinematch-tagline {
    font-size: 12px;
    color: var(--netflix-muted);
    font-style: italic;
    letter-spacing: 0.4px;
}

.cinematch-nav {
    display: flex;
    gap: 28px;
    color: var(--netflix-muted);
    font-size: 14px;
    font-weight: 500;
}

/* ---------- Hero ---------- */
.hero-banner {
    background: linear-gradient(135deg, #E50914 0%, #7a0009 100%);
    padding: 60px 40px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 40px;
    box-shadow: 0 10px 40px rgba(229,9,20,0.3);
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    color: var(--netflix-white);
    margin: 0;
    text-shadow: 2px 2px 12px rgba(0,0,0,0.5);
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 17px;
    color: rgba(255,255,255,0.94);
    margin-top: 14px;
    letter-spacing: 0.3px;
}

/* ---------- Inputs ---------- */
.stTextInput > div > div > input {
    background-color: rgba(0,0,0,0.75);
    border: 1px solid var(--netflix-gray);
    color: var(--netflix-white) !important;
    border-radius: 4px;
    padding: 14px 16px;
    font-size: 16px;
}

.stTextInput > div > div > input::placeholder {
    color: #777;
}

.stTextInput > div > div > input:focus {
    border-color: var(--netflix-red);
    outline: none;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.35);
}

/* ---------- Select ---------- */
.stSelectbox > div > div {
    background-color: rgba(0,0,0,0.75) !important;
    border: 1px solid var(--netflix-gray) !important;
    border-radius: 4px !important;
    color: var(--netflix-white) !important;
}

/* ---------- Button ---------- */
.stButton > button {
    background-color: var(--netflix-red);
    color: var(--netflix-white);
    border: none;
    border-radius: 4px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.6px;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    background-color: var(--netflix-red-hover);
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(229,9,20,0.55);
}

/* ---------- Movie Cards ---------- */
.movie-card {
    background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
    border-radius: 8px;
    padding: 22px 26px;
    margin: 12px 0;
    border-left: 4px solid var(--netflix-red);
    transition: all 0.3s ease;
    display: flex;
    gap: 18px;
    align-items: flex-start;
}

.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(229,9,20,0.35);
    border-left-color: var(--netflix-red-hover);
}

.movie-poster {
    width: 110px;
    min-width: 110px;
    border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.6);
}

.movie-content {
    flex: 1;
}

.movie-title {
    color: var(--netflix-white);
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.2px;
}

.movie-meta {
    display: flex;
    align-items: center;
    gap: 18px;
    color: var(--netflix-red);
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: 0.4px;
    flex-wrap: wrap;
}

.movie-meta span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.movie-overview {
    color: var(--netflix-muted);
    font-size: 14px;
    line-height: 1.65;
}

/* ---------- Section Title ---------- */
.section-title {
    color: var(--netflix-white);
    font-size: 26px;
    font-weight: 800;
    margin: 36px 0 18px 0;
    border-left: 5px solid var(--netflix-red);
    padding-left: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.3px;
}

/* ---------- Trending Grid ---------- */
.trend-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 18px;
    margin-top: 10px;
}

.trend-card {
    background: var(--netflix-dark);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.trend-card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 30px rgba(229,9,20,0.4);
}

.trend-poster {
    width: 100%;
    display: block;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    background: #333;
}

.trend-info {
    padding: 12px 14px;
}

.trend-title {
    color: var(--netflix-white);
    font-size: 14px;
    font-weight: 700;
    margin: 0 0 6px 0;
    line-height: 1.3;
}

.trend-meta {
    color: var(--netflix-red);
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    padding: 44px 0 24px 0;
    color: var(--netflix-gray);
    font-size: 13px;
    letter-spacing: 0.3px;
}

.footer .material-icons-round {
    font-size: 16px;
    color: var(--netflix-red);
    vertical-align: middle;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""

st.markdown(NETFLIX_CSS, unsafe_allow_html=True)


# ============================================================
# ICON HELPER
# ============================================================
def icon(name: str, size: int = 22, color: str = "#E50914") -> str:
    """Return inline HTML for a Material Round icon."""
    return (
        f'<span class="material-icons-round" '
        f'style="font-size:{size}px;color:{color};">{name}</span>'
    )


# ============================================================
# TMDb API — SAFE KEY LOADING
# ============================================================
def get_tmdb_key():
    """Read the TMDb API key from Streamlit secrets. Return None if missing."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None


TMDB_API_KEY = get_tmdb_key()
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending_movies(limit: int = 12):
    """Fetch trending movies this week from TMDb (cached 1 hour)."""
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
                    if m.get("poster_path")
                    else None
                ),
                "rating": m.get("vote_average", 0),
                "release": m.get("release_date", ""),
            }
            for m in results
        ]
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_by_title(title: str, year: str = None):
    """Look up a poster URL on TMDb by movie title (cached 24 hours)."""
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
# DATA LOADING (PUBLIC CSV — NO API KEY NEEDED)
# ============================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "IgorNascAlves/data-science-primeiros-passos/"
    "master/tmdb_5000_movies.csv"
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load TMDB 5000 movies dataset from a public URL."""
    try:
        return pd.read_csv(DATASET_URL)
    except Exception as e:
        st.error(f"Could not load the movie dataset. Error: {e}")
        st.stop()


# ============================================================
# NLP ENGINE
# ============================================================
def _parse_names(text: str) -> str:
    """Convert '[{"name":"Tom Hanks"}, ...]' -> 'tom hanks ...'"""
    try:
        items = ast.literal_eval(text)
        return " ".join(i["name"] for i in items).lower()
    except Exception:
        return ""


def _clean(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


@st.cache_resource(show_spinner="Loading CineMatch engine...")
def build_engine():
    """Build TF-IDF matrix and cosine similarity."""
    df = load_data().copy()
    df = df.dropna(subset=["title", "overview"])

    df["genres"] = df["genres"].apply(_parse_names)
    df["keywords"] = df["keywords"].apply(_parse_names)
    df["overview"] = df["overview"].apply(_clean)

    df["soup"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

    keep = [
        c
        for c in [
            "id",
            "title",
            "overview",
            "genres",
            "vote_average",
            "release_date",
            "soup",
        ]
        if c in df.columns
    ]
    df = df[keep].reset_index(drop=True)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = tfidf.fit_transform(df["soup"])
    sim = cosine_similarity(matrix, matrix)

    return df, sim


def recommend(df, sim, title, n=10):
    title = title.lower().strip()
    matches = df[df["title"].str.lower() == title]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title, na=False)]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1 : n + 1]
    result = df.iloc[[i for i, _ in scores]][
        ["title", "overview", "genres", "vote_average"]
    ].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    return result


def search_titles(df, query, limit=5):
    query = query.lower()
    return (
        df[df["title"].str.lower().str.contains(query, na=False)]["title"]
        .head(limit)
        .tolist()
    )


# ============================================================
# UI — HEADER
# ============================================================
st.markdown(
    f"""
    <div class="cinematch-header">
        <div class="cinematch-brand">
            {icon('movie', 34)}
            <div>
                <div class="cinematch-logo">CINEMATCH</div>
                <div class="cinematch-tagline">YOUR NEXT FAVORITE MOVIE IS ONE MATCH AWAY</div>
            </div>
        </div>
        <div class="cinematch-nav">
            {icon('home', 20, '#B3B3B3')} HOME&nbsp;&nbsp;
            {icon('search', 20, '#B3B3B3')} SEARCH&nbsp;&nbsp;
            {icon('favorite', 20, '#B3B3B3')} WATCHLIST
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI — HERO
# ============================================================
st.markdown(
    f"""
    <div class="hero-banner">
        {icon('local_movies', 48, '#FFFFFF')}
        <h1 class="hero-title">Find Your Next Obsession</h1>
        <p class="hero-subtitle">
            Type a movie you love — we'll match you with your next favorite.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI — SEARCH + RECOMMENDATIONS
# ============================================================
df, sim = build_engine()

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Search",
        placeholder="Try: The Dark Knight, Avatar, Inception...",
        label_visibility="collapsed",
    )

with col2:
    n_results = st.selectbox(
        "Show", [5, 10, 15, 20], index=1, label_visibility="collapsed"
    )

# Live suggestions
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


# ============================================================
# UI — RECOMMENDATION RESULTS
# ============================================================
if st.button("FIND MY MATCH", use_container_width=True):
    if not query:
        st.warning("Please enter a movie title.")
    else:
        with st.spinner("Finding your matches..."):
            results = recommend(df, sim, query, n=n_results)

        if results.empty:
            st.error(f"No movies found matching “{query}”. Try another title.")
        else:
            st.markdown(
                f"""
                <div class="section-title">
                    {icon('auto_awesome', 26)}
                    Because you liked “{query.title()}”
                </div>
                """,
                unsafe_allow_html=True,
            )

            for _, row in results.iterrows():
                rating = row.get("vote_average", None)
                rating_txt = (
                    f"{rating:.1f}/10"
                    if isinstance(rating, (int, float)) and rating
                    else "N/A"
                )
                sim_val = row["similarity"]

                # Poster from TMDb (cached)
                poster = (
                    fetch_poster_by_title(row["title"]) if TMDB_API_KEY else None
                )
                poster_html = (
                    f'<img class="movie-poster" src="{poster}" alt="poster">'
                    if poster
                    else (
                        f'<div class="movie-poster" '
                        f'style="display:flex;align-items:center;'
                        f'justify-content:center;background:#1a1a1a;">'
                        f'{icon("movie", 40, "#564d4d")}</div>'
                    )
                )

                st.markdown(
                    f"""
                    <div class="movie-card">
                        {poster_html}
                        <div class="movie-content">
                            <div class="movie-title">
                                {icon('movie', 22)} {row['title']}
                            </div>
                            <div class="movie-meta">
                                <span>{icon('star', 16)} {rating_txt}</span>
                                <span>{icon('trending_up', 16)} Similarity: {sim_val:.2f}</span>
                            </div>
                            <div class="movie-overview">
                                {str(row['overview'])[:320]}...
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# UI — TRENDING NOW (TMDb API)
# ============================================================
if TMDB_API_KEY:
    st.markdown(
        f"""
        <div class="section-title">
            {icon('local_fire_department', 26)}
            Trending This Week
        </div>
        """,
        unsafe_allow_html=True,
    )

    trending = fetch_trending_movies(12)

    if not trending:
        st.info("Trending movies are unavailable right now. Please try again later.")
    else:
        cards_html = []
        for m in trending:
            poster = m["poster"] or ""
            rating = m["rating"] or 0
            release = (m["release"] or "")[:4]

            poster_tag = (
                f'<img class="trend-poster" src="{poster}" alt="poster">'
                if poster
                else (
                    f'<div class="trend-poster" '
                    f'style="display:flex;align-items:center;justify-content:center;">'
                    f'{icon("movie", 40, "#564d4d")}</div>'
                )
            )

            cards_html.append(
                f"""
                <div class="trend-card">
                    {poster_tag}
                    <div class="trend-info">
                        <div class="trend-title">{m['title']}</div>
                        <div class="trend-meta">
                            {icon('star', 14)} {rating:.1f}
                            &nbsp; {release}
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
    st.markdown(
        f"""
        <div style="margin-top:40px;padding:20px;border-radius:8px;
                    background:#1a1a1a;border-left:4px solid #E50914;
                    color:#B3B3B3;font-size:14px;">
            {icon('info', 20)} Trending movies require a TMDb API key.
            Add <code>TMDB_API_KEY</code> to your Streamlit Secrets to enable this section.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UI — FOOTER
# ============================================================
st.markdown(
    f"""
    <div class="footer">
        CINEMATCH &nbsp;•&nbsp; Powered by NLP & TF-IDF &nbsp;•&nbsp;
        Built with {icon('favorite', 16)} at TekHer AI Bootcamp
    </div>
    """,
    unsafe_allow_html=True,
)