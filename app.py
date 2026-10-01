"""
CineMatch — Netflix-style movie recommender
Streamlit Cloud-ready with TMDb API integration.
Single-page app with top navigation (no sidebar).
Uses dense embeddings (LSA via TruncatedSVD) — no user-facing toggle.

Cache note: poster-fetching functions use a versioned parameter (`v`)
so that bumping the version invalidates any stale "no poster" results
cached while the TMDb API key was still invalid.
"""

import ast
import datetime
import random
import textwrap
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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
    "page": "home",
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
# CSS — NETFLIX DESIGN
# ============================================================
CSS = """
:root {
    --nf-red: #E50914;
    --nf-red-hover: #F40612;
    --nf-black: #141414;
    --nf-dark: #181818;
    --nf-card: #1f1f1f;
    --nf-border: #2a2a2a;
    --nf-white: #FFFFFF;
    --nf-gray: #b3b3b3;
    --nf-gray-dark: #808080;
    --nf-gold: #f5c518;
}
html, body, .stApp {
    background-color: var(--nf-black) !important;
    color: var(--nf-white);
    font-family: 'Inter', -apple-system, 'Helvetica Neue', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* ============ HEADER ============ */
.nf-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 8px 14px 8px;
}
.nf-brand { display: flex; align-items: center; gap: 8px; }
.nf-brand-logo {
    font-size: 26px;
    font-weight: 900;
    color: var(--nf-red);
    letter-spacing: -1.2px;
    text-transform: uppercase;
    line-height: 1;
}
.nf-header-icons {
    display: flex;
    align-items: center;
    gap: 20px;
    color: var(--nf-white);
}
.nf-avatar {
    width: 32px; height: 32px; border-radius: 4px;
    background: linear-gradient(135deg, #E50914 0%, #7a0009 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 14px;
}

/* ============ NAV ============ */
.nf-nav {
    display: flex;
    align-items: center;
    gap: 26px;
    padding: 6px 8px 14px 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 28px;
}
.nf-nav .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: var(--nf-gray) !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    padding: 4px 0 !important;
    margin: 0 !important;
    width: auto !important;
    min-width: 0 !important;
    box-shadow: none !important;
    text-align: left !important;
    letter-spacing: 0.2px !important;
}
.nf-nav .stButton > button:hover {
    color: var(--nf-white) !important;
    background: transparent !important;
}
.nf-nav .stButton > button[kind="primary"] {
    color: var(--nf-white) !important;
    font-weight: 700 !important;
    border-bottom: 3px solid var(--nf-red) !important;
    padding-bottom: 6px !important;
}

/* ============ HERO ============ */
.nf-hero {
    background: linear-gradient(90deg,
        rgba(20,20,20,0.95) 0%,
        rgba(20,20,20,0.75) 40%,
        rgba(20,20,20,0.3) 70%,
        rgba(20,20,20,0.6) 100%),
        linear-gradient(135deg, #2a0a0a 0%, #141414 60%);
    border-radius: 14px;
    padding: 56px 48px;
    margin-bottom: 36px;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(229,9,20,0.15);
}
.nf-hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(229,9,20,0.15);
    border: 1px solid rgba(229,9,20,0.4);
    color: var(--nf-red);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11.5px; font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 16px;
    width: fit-content;
}
.nf-hero-title {
    font-size: 52px; font-weight: 900;
    color: var(--nf-white);
    margin: 0 0 14px 0;
    line-height: 1.05;
    letter-spacing: -1.4px;
    max-width: 620px;
}
.nf-hero-sub {
    font-size: 17px;
    color: rgba(255,255,255,0.85);
    max-width: 520px;
    line-height: 1.55;
    margin: 0;
}

/* ============ SECTIONS ============ */
.nf-section {
    display: flex; align-items: center; gap: 10px;
    font-size: 23px; font-weight: 800;
    color: var(--nf-white);
    margin: 36px 0 16px 0;
    letter-spacing: -0.3px;
}
.nf-section-accent {
    width: 4px; height: 22px;
    background: var(--nf-red);
    border-radius: 2px;
}

/* ============ GENRE CARDS ============ */
.nf-genre-card {
    position: relative;
    height: 160px;
    border-radius: 8px;
    overflow: hidden;
    background-size: cover;
    background-position: center;
    border: 2px solid transparent;
    transition: transform 0.22s ease, border-color 0.22s ease;
    display: flex;
    align-items: flex-end;
    padding: 16px;
    color: white;
    margin-bottom: 4px;
}
.nf-genre-card::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(to top,
        rgba(0,0,0,0.9) 8%,
        rgba(0,0,0,0.2) 60%,
        rgba(0,0,0,0.1) 100%);
    z-index: 1;
}
.nf-genre-card:hover {
    transform: scale(1.02);
    border-color: var(--nf-red);
}
.nf-genre-content { position: relative; z-index: 2; width: 100%; }
.nf-genre-icon { margin-bottom: 8px; display: block; }
.nf-genre-name {
    font-size: 19px; font-weight: 800;
    margin: 0; letter-spacing: -0.2px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.6);
}
.nf-genre-count {
    font-size: 12.5px;
    color: rgba(255,255,255,0.75);
    margin-top: 3px;
    display: flex; align-items: center; gap: 6px;
}
.nf-genre-count .dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--nf-red); display: inline-block;
}

/* ============ RESULT CARDS ============ */
.nf-result-card {
    position: relative;
    background: var(--nf-card);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.04);
}
.nf-result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
    border-color: rgba(229,9,20,0.4);
}
.nf-result-poster-wrap { position: relative; overflow: hidden; }
.nf-result-poster {
    width: 100%; aspect-ratio: 16 / 10;
    object-fit: cover; display: block;
    background: #111;
    transition: transform 0.35s ease;
}
.nf-result-card:hover .nf-result-poster { transform: scale(1.05); }
.nf-result-overlay {
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(6px);
    color: var(--nf-gold);
    padding: 4px 8px; border-radius: 4px;
    font-size: 11.5px; font-weight: 800;
    display: flex; align-items: center; gap: 3px;
}
.nf-result-body { padding: 14px 15px 16px; min-height: 118px; }
.nf-result-title {
    font-size: 15px; font-weight: 800;
    color: var(--nf-white);
    margin: 0 0 3px 0; line-height: 1.25;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.nf-result-meta {
    display: flex; align-items: center; gap: 10px;
    font-size: 12.5px; font-weight: 700;
    margin-bottom: 8px;
}
.nf-result-year { color: var(--nf-gray-dark); font-weight: 600; }
.nf-result-match {
    color: var(--nf-red);
    display: inline-flex; align-items: center; gap: 3px;
}
.nf-result-overview {
    font-size: 12.5px; color: var(--nf-gray);
    line-height: 1.5;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
}

/* ============ SHELF ============ */
.nf-shelf {
    display: flex; gap: 12px;
    overflow-x: auto;
    padding-bottom: 12px; margin-bottom: 8px;
    scrollbar-width: thin;
    scrollbar-color: #333 transparent;
}
.nf-shelf::-webkit-scrollbar { height: 6px; }
.nf-shelf::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }
.nf-shelf::-webkit-scrollbar-thumb:hover { background: var(--nf-red); }
.nf-shelf-card {
    flex: 0 0 155px;
    background: var(--nf-card);
    border-radius: 8px; overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(255,255,255,0.04);
}
.nf-shelf-card:hover {
    transform: scale(1.06);
    box-shadow: 0 12px 30px rgba(229,9,20,0.25);
    border-color: rgba(229,9,20,0.5);
}
.nf-shelf-poster {
    width: 100%; aspect-ratio: 2 / 3;
    object-fit: cover; background: #111;
    display: block;
}
.nf-shelf-info { padding: 10px 11px 12px; }
.nf-shelf-title {
    font-size: 12.5px; font-weight: 700;
    color: var(--nf-white);
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; margin-bottom: 4px;
}
.nf-shelf-meta {
    font-size: 11.5px; color: var(--nf-gold);
    font-weight: 700;
    display: flex; align-items: center; gap: 4px;
}

/* ============ MOTD ============ */
.nf-motd {
    display: flex; gap: 26px;
    background: linear-gradient(135deg, #1f1f1f 0%, #161616 100%);
    border-radius: 12px; padding: 26px;
    margin-bottom: 12px;
    border-left: 4px solid var(--nf-red);
    border-top: 1px solid rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.nf-motd-poster {
    width: 130px; min-width: 130px; height: 195px;
    object-fit: cover; border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.nf-motd-label {
    color: var(--nf-red);
    font-weight: 800; font-size: 11.5px;
    letter-spacing: 1.4px; text-transform: uppercase;
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}
.nf-motd-title {
    font-size: 28px; font-weight: 900;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px; line-height: 1.15;
}
.nf-motd-meta {
    display: flex; align-items: center; gap: 12px;
    color: var(--nf-gold); font-weight: 700;
    font-size: 13px; margin-bottom: 12px;
}
.nf-motd-overview {
    color: var(--nf-gray); font-size: 14px;
    line-height: 1.6;
}

/* ============ WATCHLIST ============ */
.nf-watch-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 18px;
    margin-bottom: 8px;
}
.nf-watch-card {
    position: relative;
    transition: transform 0.22s ease;
}
.nf-watch-card:hover { transform: translateY(-4px); }
.nf-watch-poster-wrap {
    position: relative; border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.nf-watch-poster {
    width: 100%; aspect-ratio: 2 / 3;
    object-fit: cover; display: block;
    background: #111;
}
.nf-watch-rating {
    position: absolute; top: 10px; left: 10px;
    background: rgba(20,20,20,0.92);
    backdrop-filter: blur(6px);
    border-radius: 5px; padding: 4px 8px;
    font-size: 12px; font-weight: 800;
    color: var(--nf-gold);
    display: flex; align-items: center; gap: 3px;
}
.nf-watch-remove {
    position: absolute; top: 10px; right: 10px;
    width: 26px; height: 26px; border-radius: 50%;
    background: rgba(229,9,20,0.95);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 900; font-size: 14px;
}
.nf-watch-title {
    font-size: 13.5px; font-weight: 800;
    color: var(--nf-white);
    margin: 12px 0 3px 0; line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.nf-watch-year {
    font-size: 12px;
    color: var(--nf-gray-dark);
}

/* ============ EMPTY STATE ============ */
.nf-empty {
    padding: 64px 24px; border-radius: 12px;
    background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
    border: 1px solid var(--nf-border);
    text-align: center; color: var(--nf-gray);
}
.nf-empty-icon {
    color: var(--nf-gray-dark);
    opacity: 0.5; margin-bottom: 18px;
    display: inline-block;
}
.nf-empty-text {
    font-size: 15px; line-height: 1.6;
    max-width: 400px; margin: 0 auto;
}

/* ============ BUTTONS ============ */
.stButton > button {
    background-color: var(--nf-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 11px 20px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover { background-color: var(--nf-red-hover) !important; }

/* ============ INPUTS ============ */
.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--nf-red) !important;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.2) !important;
}
.stTextInput > div > div > input::placeholder { color: #666 !important; }
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: white !important;
}

/* ============ FOOTER ============ */
.nf-footer {
    text-align: center;
    padding: 48px 0 12px;
    color: var(--nf-gray-dark);
    font-size: 12px;
}
.nf-footer-accent { color: var(--nf-red); font-weight: 700; }

/* ============ RESPONSIVE ============ */
@media (max-width: 900px) {
    .nf-hero { padding: 40px 28px; min-height: 220px; }
    .nf-hero-title { font-size: 38px; letter-spacing: -1px; }
    .nf-hero-sub { font-size: 15px; }
    .nf-brand-logo { font-size: 22px; }
    .nf-nav { gap: 18px; }
    .nf-section { font-size: 20px; }
}
@media (max-width: 600px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    .nf-hero { padding: 32px 20px; border-radius: 10px; }
    .nf-hero-title { font-size: 28px; }
    .nf-hero-sub { font-size: 13.5px; }
    .nf-brand-logo { font-size: 20px; }
    .nf-header-icons { gap: 12px; }
    .nf-nav { gap: 14px; padding: 6px 4px 12px 4px; }
    .nf-nav .stButton > button { font-size: 12.5px !important; }
    .nf-section { font-size: 18px; margin: 26px 0 12px 0; }
    .nf-motd { flex-direction: column; padding: 20px; }
    .nf-motd-poster { width: 100%; height: auto; max-width: 240px; }
    .nf-motd-title { font-size: 22px; }
    .nf-genre-card { height: 130px; }
    .nf-genre-name { font-size: 15px; }
    .nf-shelf-card { flex: 0 0 125px; }
    .nf-watch-grid { grid-template-columns: repeat(2, 1fr); gap: 12px; }
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
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
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
    "bell": '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>',
}

FILLED_ICONS = {"star", "favorite", "bolt", "auto_awesome", "bell"}


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
# TMDb API — with cache-busting version parameter
# ============================================================
CACHE_VERSION = 3   # bump to invalidate stale poster caches


def get_tmdb_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None


TMDB_API_KEY = get_tmdb_key()
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/w780"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending_movies(limit: int = 14, v: int = CACHE_VERSION):
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
def fetch_poster_by_id(movie_id, v: int = CACHE_VERSION):
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
def fetch_poster_by_title(title: str, v: int = CACHE_VERSION):
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
def fetch_genre_backdrop(genre_name: str, v: int = CACHE_VERSION):
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


def get_poster(movie_id=None, title=None) -> str | None:
    """Try id-based lookup first (more reliable), then title fallback."""
    p = fetch_poster_by_id(movie_id) if movie_id else None
    if not p and title:
        p = fetch_poster_by_title(title)
    return p


def placeholder_bg(seed_text: str) -> str:
    palettes = [
        "#3a0a0a,#120202", "#0a1a3a,#020712", "#1a0a3a,#070212",
        "#0a3a1a,#021207", "#3a1a0a,#120702",
    ]
    idx = sum(ord(c) for c in seed_text) % len(palettes)
    a, b = palettes[idx].split(",")
    return f"linear-gradient(135deg, {a}, {b})"


# ============================================================
# DATA + EMBEDDINGS ENGINE (LSA)
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
    """
    Build dense-vector embeddings using LSA (TruncatedSVD).
      1. TF-IDF vectorize each movie's text.
      2. Compress sparse matrix → 200 dense dimensions (captures semantics).
      3. Normalize so cosine similarity == dot product.
    """
    df = load_data().copy()
    df = df.dropna(subset=["title", "overview"])
    df["genres_list"] = df["genres"].apply(_parse_names_titlecase)
    df["genres"] = df["genres"].apply(_parse_names)
    df["keywords"] = df["keywords"].apply(_parse_names)
    df["overview"] = df["overview"].apply(_clean)
    df["soup"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]

    keep = [
        c for c in [
            "id", "title", "overview", "genres", "genres_list",
            "vote_average", "release_date", "soup",
        ] if c in df.columns
    ]
    df = df[keep].reset_index(drop=True)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    n_components = min(200, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(tfidf_matrix)

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1
    dense_norm = dense / norms
    sim = dense_norm @ dense_norm.T

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
    result = df.iloc[indices][
        ["id", "title", "overview", "genres", "genres_list", "vote_average", "release_date"]
    ].copy()
    result["similarity"] = [round(float(s), 3) for _, s in scores]
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
# HEADER
# ============================================================
render_html(f"""
<div class="nf-header">
    <div class="nf-brand">
        <div class="nf-brand-logo">CineMatch</div>
    </div>
    <div class="nf-header-icons">
        {icon('search', 20, '#fff')}
        {icon('bell', 20, '#fff')}
        <div class="nf-avatar">JU</div>
    </div>
</div>
""")


# ============================================================
# NAV — native Streamlit buttons styled as Netflix nav
# ============================================================
NAV_ITEMS = [
    ("home", "Home"),
    ("browse", "Browse"),
    ("search", "Search"),
    ("watchlist", "My List"),
]

nav_cols = st.columns([1, 1, 1, 1, 5])
for i, (page_key, label) in enumerate(NAV_ITEMS):
    with nav_cols[i]:
        is_active = st.session_state.page == page_key
        if st.button(
            label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = page_key
            st.session_state.selected_genre = None
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
        f'<img class="nf-shelf-poster" src="{poster}" alt="{title}">'
        if poster else
        f'<div class="nf-shelf-poster" style="background:{placeholder_bg(title)};'
        f'display:flex;align-items:center;justify-content:center;">'
        f'{icon("movie", 30, "rgba(255,255,255,0.4)")}</div>'
    )
    year_txt = f" · {year}" if year else ""
    return f"""
    <div class="nf-shelf-card">
        {poster_html}
        <div class="nf-shelf-info">
            <div class="nf-shelf-title">{title}</div>
            <div class="nf-shelf-meta">{icon('star', 11, 'var(--nf-gold)')} {rating:.1f}{year_txt}</div>
        </div>
    </div>
    """


def render_result_card(row, show_similarity=True):
    movie_id = row.get("id")
    rating = row.get("vote_average") or 0
    sim_val = row.get("similarity")
    year = str(row.get("release_date") or "")[:4]
    overview = str(row.get("overview", ""))[:140]

    poster = get_poster(movie_id=movie_id, title=row["title"])

    poster_html = (
        f'<img class="nf-result-poster" src="{poster}" alt="{row["title"]}">'
        if poster else
        f'<div class="nf-result-poster" style="background:{placeholder_bg(row["title"])};'
        f'display:flex;align-items:center;justify-content:center;">'
        f'{icon("movie", 34, "rgba(255,255,255,0.4)")}</div>'
    )

    match_html = ""
    if show_similarity and sim_val is not None:
        pct = min(99, int(sim_val * 100))
        match_html = f'<span class="nf-result-match">{pct}% match</span>'

    render_html(f"""
        <div class="nf-result-card">
            <div class="nf-result-poster-wrap">
                {poster_html}
                <div class="nf-result-overlay">{icon('star', 11, 'var(--nf-gold)')} {rating:.1f}</div>
            </div>
            <div class="nf-result-body">
                <div class="nf-result-title">{row['title']}</div>
                <div class="nf-result-meta">
                    <span class="nf-result-year">{year}</span>
                    {match_html}
                </div>
                <div class="nf-result-overview">{overview}</div>
            </div>
        </div>
    """)

    if movie_id is not None:
        in_wl = is_in_watchlist(movie_id)
        label = "♥  In Watchlist" if in_wl else "+  Add to Watchlist"
        if st.button(label, key=f"wl_{movie_id}_{row['title'][:20]}", use_container_width=True):
            if in_wl:
                remove_from_watchlist(movie_id)
            else:
                add_to_watchlist(
                    movie_id, row["title"], poster,
                    f"{rating:.1f}", year, row.get("genres_list"),
                )
            st.rerun()


# ============================================================
# PAGE: HOME
# ============================================================
def render_home():
    render_html(f"""
        <div class="nf-hero">
            <div class="nf-hero-badge">
                {icon('auto_awesome', 12, 'var(--nf-red)')} AI-POWERED · NLP
            </div>
            <h1 class="nf-hero-title">Find Your Next Obsession</h1>
            <p class="nf-hero-sub">Discover movies that match your mood. Powered by semantic embeddings — not just keywords.</p>
        </div>
    """)

    render_html(
        f'<div class="nf-section"><span class="nf-section-accent"></span>'
        f'{icon("today", 20, "var(--nf-red)")} Movie of the Day</div>'
    )

    seed = int(datetime.date.today().strftime("%Y%m%d"))
    random.seed(seed)
    top = df[df["vote_average"] >= 7.5] if "vote_average" in df.columns else df
    if not top.empty:
        motd = top.sample(1).iloc[0]
        poster = get_poster(movie_id=motd.get("id"), title=motd["title"])
        poster_html = (
            f'<img class="nf-motd-poster" src="{poster}" alt="poster">'
            if poster else
            f'<div class="nf-motd-poster" style="background:{placeholder_bg(motd["title"])};'
            f'display:flex;align-items:center;justify-content:center;">'
            f'{icon("movie", 32, "rgba(255,255,255,0.4)")}</div>'
        )
        rating = motd.get("vote_average", 0)
        year = str(motd.get("release_date") or "")[:4]
        render_html(f"""
            <div class="nf-motd">
                {poster_html}
                <div style="flex:1;min-width:0;">
                    <div class="nf-motd-label">
                        {icon('auto_awesome', 12, 'var(--nf-red)')} PICKED FOR TODAY
                    </div>
                    <div class="nf-motd-title">{motd['title']}</div>
                    <div class="nf-motd-meta">
                        {icon('star', 14, 'var(--nf-gold)')} {rating:.1f}
                        <span style="color: var(--nf-gray-dark);font-weight:600;">· {year}</span>
                    </div>
                    <div class="nf-motd-overview">{str(motd['overview'])[:340]}…</div>
                </div>
            </div>
        """)

    if TMDB_API_KEY:
        render_html(
            f'<div class="nf-section"><span class="nf-section-accent"></span>'
            f'{icon("local_fire_department", 20, "var(--nf-red)")} Trending This Week</div>'
        )
        trending = fetch_trending_movies(14)
        if trending:
            cards = "".join([
                render_shelf_card(m["title"], m["poster"], m["rating"] or 0, m.get("release"))
                for m in trending
            ])
            render_html(f'<div class="nf-shelf">{cards}</div>')

    render_html(
        f'<div class="nf-section"><span class="nf-section-accent"></span>'
        f'{icon("explore", 20, "var(--nf-red)")} Explore by Genre</div>'
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
        render_html(
            f'<div class="nf-section"><span class="nf-section-accent"></span>'
            f'{icon(GENRE_ICONS.get(genre, "movie"), 20, "var(--nf-red)")} {label}</div>'
        )

        movies = get_genre_movies(df, genre, n=20)
        if movies.empty:
            st.info(f"No movies found in {label}.")
        else:
            st.caption(f"Top {len(movies)} highest-rated films in {label}")
            cols = st.columns(4)
            for i, (_, row) in enumerate(movies.iterrows()):
                with cols[i % 4]:
                    render_result_card(row, show_similarity=False)
        return

    render_html("""
        <div class="nf-hero" style="min-height:180px;padding:40px 32px;">
            <div class="nf-hero-badge">🎬  DISCOVER</div>
            <h1 class="nf-hero-title" style="font-size:40px;">Browse by Genre</h1>
            <p class="nf-hero-sub">Don't know what to watch? Pick a mood and we'll show you the best.</p>
        </div>
    """)

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
                    <div class="nf-genre-card" style="{bg_style}">
                        <div class="nf-genre-content">
                            <div class="nf-genre-icon">{icon(icon_name, 28, 'white')}</div>
                            <div class="nf-genre-name">{label}</div>
                            <div class="nf-genre-count"><span class="dot"></span>{count:,} movies</div>
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
    col_input, col_btn = st.columns([9, 1.2])
    with col_input:
        query = st.text_input(
            "Search", placeholder="Try: The Dark Knight, Inception, Avatar…",
            label_visibility="collapsed", key="search_query",
        )
    with col_btn:
        go = st.button("Search", key="search_go_btn", use_container_width=True, type="primary")

    if st.session_state.history:
        hist = "  ·  ".join(st.session_state.history)
        render_html(
            f'<div style="color:#888;font-size:12.5px;margin:-6px 0 18px 2px;">'
            f'{icon("history", 14, "#888")} Recent: {hist}</div>'
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
            render_html(
                f'<div style="color:#888;font-size:12.5px;margin:2px 0 18px 2px;">'
                f'{icon("lightbulb", 14, "#888")} Suggestions: {"  ·  ".join(suggestions)}</div>'
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

            with st.spinner("Finding matches…"):
                results = recommend(df, sim, query, n=8, genre_filter=st.session_state.genre_filter)

            if results.empty:
                st.error(f'No movies found matching "{query}". Try another title.')
            else:
                render_html(
                    f'<div class="nf-section" style="margin-top:28px;">'
                    f'<span class="nf-section-accent"></span>'
                    f'Because you liked <em style="color:var(--nf-red);font-style:italic;">"{query.title()}"</em>'
                    f'</div>'
                )
                cols = st.columns(4)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 4]:
                        render_result_card(row, show_similarity=True)


# ============================================================
# PAGE: WATCHLIST
# ============================================================
def render_watchlist():
    render_html(f"""
        <div class="nf-section" style="margin-top:0;">
            <span class="nf-section-accent"></span>
            {icon('favorite', 22, 'var(--nf-red)')} My Watchlist
        </div>
    """)

    n = len(st.session_state.watchlist)
    render_html(
        f'<div style="color:var(--nf-gray);font-size:14px;margin-bottom:20px;">'
        f'{n} title{"s" if n != 1 else ""} · sorted by added date</div>'
    )

    if not st.session_state.watchlist:
        render_html(f"""
            <div class="nf-empty">
                <div class="nf-empty-icon">{icon('movie', 56, '#444')}</div>
                <p class="nf-empty-text">
                    Your watchlist is empty.<br>
                    Search for movies and tap <strong style="color:var(--nf-red);">+ Add to Watchlist</strong> to save them here.
                </p>
            </div>
        """)
        return

    cards = []
    for m in st.session_state.watchlist:
        poster = m.get("poster") or get_poster(movie_id=m["id"], title=m["title"])
        poster_html = (
            f'<img class="nf-watch-poster" src="{poster}" alt="{m["title"]}">'
            if poster else
            f'<div class="nf-watch-poster" style="background:{placeholder_bg(m["title"])};'
            f'display:flex;align-items:center;justify-content:center;">'
            f'{icon("movie", 28, "rgba(255,255,255,0.4)")}</div>'
        )
        year = m.get("year") or ""
        genres_txt = ", ".join((m.get("genres") or [])[:2])
        sub = " · ".join([x for x in [year, genres_txt] if x])

        cards.append(f"""
            <div class="nf-watch-card">
                <div class="nf-watch-poster-wrap">
                    {poster_html}
                    <div class="nf-watch-rating">{icon('star', 11, 'var(--nf-gold)')} {m.get('rating') or 'N/A'}</div>
                    <div class="nf-watch-remove">×</div>
                </div>
                <div class="nf-watch-title">{m['title']}</div>
                <div class="nf-watch-year">{sub}</div>
            </div>
        """)

    render_html(f'<div class="nf-watch-grid">{"".join(cards)}</div>')

    st.markdown("######")
    remove_cols = st.columns(len(st.session_state.watchlist))
    for i, m in enumerate(st.session_state.watchlist):
        with remove_cols[i]:
            if st.button("Remove", key=f"rm_{m['id']}"):
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
PAGES.get(st.session_state.page, render_home)()


# ============================================================
# FOOTER
# ============================================================
render_html(f"""
    <div class="nf-footer">
        <span class="nf-footer-accent">CINEMATCH</span> &nbsp;·&nbsp;
        Powered by semantic embeddings &nbsp;·&nbsp;
        Built with {icon('favorite', 11, 'var(--nf-red)')} by Justine Umutoni © 2026
    </div>
""")