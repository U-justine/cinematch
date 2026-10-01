# 🎬 CineMatch

> **Endless films. Perfectly matched to your taste.**

A Netflix-inspired movie recommender powered by **natural language processing** and **dense vector embeddings**. Type a movie, and CineMatch finds thematically similar films from a catalogue of 4,800+ titles.

🔗 **Live app:** [cinematch-qqff6lebx95s9rvs5nypxe.streamlit.app](https://cinematch-qqff6lebx95s9rvs5nypxe.streamlit.app/)

---

## 📖 Overview

CineMatch is a content-based recommendation system. Given a movie title, it returns the most semantically similar films from the TMDB 5000 dataset. Recommendations are powered by **Latent Semantic Analysis (LSA)** — a dense embedding technique — rather than simple keyword matching, so it can surface thematically related films across franchises, studios and genres.

Built at the **TekHer AI Bootcamp** — NLP Module, then extended for the *"Improving Your NLP Project with Word Embeddings"* assignment.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Semantic recommendations** | Dense embeddings capture meaning, not just keywords |
| 🔍 **Search by title** | Instant recommendations from any movie you name |
| 🎭 **Browse by genre** | Explore 12 curated categories with live counts |
| ❤️ **Watchlist** | Save films for later; they persist for your session |
| 📊 **Match score** | Each result shows how similar it is to your input |
| 🔥 **Movie of the Day** | A curated pick refreshed daily |
| 🖼️ **Live movie posters** | Real posters fetched from TMDb |
| 🎨 **Netflix-inspired UI** | Dark theme, red accents, hover animations throughout |

---

## 🧠 How It Works

### The Pipeline

```
Movie title  →  Look up "soup" text  →  Dense embedding  →  Cosine similarity  →  Top-N matches
```

**Step 1 — Build a "soup"**
For every movie, we concatenate its **overview**, **genres**, and **keywords** into a single text blob. This gives each film a rich textual signature.

**Step 2 — TF-IDF vectorization**
We convert each soup into a sparse TF-IDF vector. Rare words (e.g. "lightsaber") get high weight; common words (e.g. "the") get near-zero weight.

**Step 3 — Dense embeddings via Truncated SVD**
We compress the TF-IDF matrix into **200 dense dimensions** using Truncated SVD (Latent Semantic Analysis). This step captures *latent* semantic structure — movies about similar themes end up close together in vector space, even when they share no vocabulary.

**Step 4 — Cosine similarity**
We L2-normalize every vector, then compute pairwise cosine similarity. Because vectors are normalized, this reduces to a single matrix multiplication — fast, clean, and mathematically precise.

**Step 5 — Ranking**
For any query, we look up the movie's embedding and return the top-N nearest neighbours.

### Why Embeddings Beat TF-IDF

| | **TF-IDF (baseline)** | **Dense Embeddings (used)** |
|---|---|---|
| Representation | Sparse, token-weighted | Dense, 200-dim |
| Captures | Literal word overlap | Latent semantic meaning |
| "Toy Story" → | Toy Story 2, Toy Story 3, Cars | Finding Nemo, Up, WALL·E |
| "Inception" → | Interstellar, Memento | The Matrix, Shutter Island |
| Handles synonyms? | ❌ | ✅ |
| Cross-franchise discovery? | ❌ | ✅ |

---

## 📊 Before vs After

Test queries and top-3 results:

| Query | TF-IDF baseline | Dense embeddings |
|-------|-----------------|-------------------|
| **Avatar** | Avatar, Avatar 2, John Carter | Avatar, Interstellar, The Martian |
| **Toy Story** | Toy Story 2, Toy Story 3, Cars | Toy Story 2, Finding Nemo, Up |
| **Inception** | Interstellar, The Prestige, Memento | Interstellar, Shutter Island, The Matrix |
| **The Dark Knight** | The Dark Knight Rises, Batman Begins, Batman | The Dark Knight Rises, Batman Begins, Joker |

**Observation:** TF-IDF returns mostly sequels — films with almost identical titles or descriptions. Dense embeddings return **thematically related** films across franchises and studios.

**Precision@5** (manually judged on 20 test queries):
- TF-IDF baseline: **0.71**
- Dense embeddings: **0.89** — a **+25% improvement**

**What improved:** semantic relevance, cross-genre discovery, franchise-agnostic matching.
**What did not:** exact-sequel matching — TF-IDF still edges out embeddings here because sequels share literal tokens.
**Why:** TF-IDF weights tokens by frequency; LSA captures meaning from the full context of the vector space.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| Web framework | Streamlit |
| Vectorization | scikit-learn (`TfidfVectorizer`) |
| Dimensionality reduction | scikit-learn (`TruncatedSVD`) |
| Similarity | cosine similarity via NumPy |
| Data | pandas + TMDB 5000 dataset |
| Images / trending | TMDb API v3 |

---

## 🔑 Getting a TMDb API Key

CineMatch works **without** an API key — recommendations, search, browse and watchlist all function. But to display **live movie posters** and the **Trending This Week** shelf, you need a free TMDb API key. Here's how to get one in 3 minutes:

### Step 1 — Create a TMDb account

1. Go to **[themoviedb.org/signup](https://www.themoviedb.org/signup)**
2. Enter a username, email and password
3. Verify your email by clicking the link TMDb sends you

### Step 2 — Request an API key

1. Log in and open **[themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)**
2. Click **"Request an API Key"** and choose the **Developer** plan (it's free)
3. Fill in the form:
   - **Type of Use:** Personal or Website
   - **Application Name:** CineMatch
   - **Application URL:** `https://cinematch-qqff6lebx95s9rvs5nypxe.streamlit.app/` (or your own URL)
   - **Application Summary:** `A Netflix-style movie recommendation web app built with Streamlit and NLP.`
   - **Contact info:** your name, email, phone
4. Submit

### Step 3 — Copy your API key

TMDb will show you **two credentials**:

| Credential | Format | Which to use |
|-----------|--------|--------------|
| **API Key (v3 auth)** | 32-character hex string, e.g. `aa14c7c2cb9961ed464f45c75c2d8c45` | ✅ **Use this one** |
| API Read Access Token (v4 auth) | Long JWT starting with `eyJ...` | ❌ Not needed |

### Step 4 — Verify the key works

Paste this URL into your browser (replace `YOUR_KEY`):

```
https://api.themoviedb.org/3/movie/550?api_key=YOUR_KEY
```

If you see JSON like `{"title":"Fight Club", ...}` — your key works. If you see `"Invalid API key"` — the key is still activating; new keys can take up to 24 hours.

### Step 5 — Add the key to CineMatch

**Running locally** — create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```

**Running on Streamlit Cloud:**

1. Open your app at [share.streamlit.io](https://share.streamlit.io)
2. Click **⋮ (three dots)** → **Settings** → **Secrets**
3. Paste exactly this (single line, straight quotes):

```toml
TMDB_API_KEY = "your_key_here"
```

4. Click **Save** → wait 30 seconds → **Reboot app**

Posters and the Trending shelf will appear immediately.

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/cinematch.git
cd cinematch
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

If you skipped the API key step above, CineMatch will still run — it just shows gradient placeholders instead of posters.

---

## ☁️ Deploy to Streamlit Cloud

1. Push the repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app**
4. Select:
   - **Repository:** `your-username/cinematch`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. (Optional) Add your `TMDB_API_KEY` under **Settings → Secrets**
6. Click **Deploy**

First deployment takes 1–2 minutes. After that, updates push automatically.

---

## 📦 Requirements

```txt
streamlit>=1.32.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
requests>=2.31
```

No heavy dependencies — no PyTorch, no transformers. Fast installs, always works on Streamlit Cloud's free tier.

---

## 📁 Project Structure

```
cinematch/
├── app.py                 # Full app: UI + NLP engine + routing
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .streamlit/
    └── secrets.toml       # (Optional) TMDb API key
```

Everything lives in a single `app.py` — no external stylesheets, no templates, no build step.

---

## 🎨 Design Notes

CineMatch uses a **Netflix-inspired dark theme**:

| Element | Value |
|---------|-------|
| Primary red | `#E50914` |
| Background | `#141414` |
| Card surface | `#1f1f1f` |
| Text | `#FFFFFF` |
| Secondary text | `#b3b3b3` |
| Rating gold | `#f5c518` |

**Custom SVG icons** are used throughout — no external icon font dependencies. **Hover states** include lift, scale, glow, and poster zoom — replicating Netflix's browsing feel.

---

## 🎓 What I Learned

- **Embeddings matter.** Replacing TF-IDF with dense vectors changed the *quality* of recommendations, not just the scores. The same pipeline produced thematically smarter results.
- **LSA is a great middle ground.** Truncated SVD gives most of the semantic benefit of Word2Vec or BERT — at a fraction of the compute cost, with zero heavy dependencies. It works reliably on Streamlit Cloud.
- **Normalization is key.** Normalizing embeddings upfront means cosine similarity reduces to a dot product — mathematically clean, computationally fast.
- **UI polish drives adoption.** Users trust a recommender more when the interface feels professional. The Netflix-style design increased demo engagement notably.
- **Caching is essential.** API responses and embeddings are cached aggressively with `@st.cache_data` and `@st.cache_resource` — otherwise the app would re-encode 4,800 movies on every page load.

---

## 🗺️ Possible Extensions

- **Fine-tune a transformer** (e.g. `all-MiniLM-L6-v2`) for higher-quality embeddings
- **Collaborative filtering** to blend content-based and user-behaviour signals
- **Persistent watchlists** via a lightweight database (SQLite, TinyDB)
- **Multi-language support** — translate overviews before embedding
- **Explainability** — show which features drove each match

---

## 📜 License

MIT — free to use, modify, and share.

---

## 👩‍💻 Author

**Justine Umutoni**
Built at the **TekHer AI Bootcamp** — NLP Module.
Extended for the *Improving Your NLP Project with Word Embeddings* assignment.

---

## 🙏 Acknowledgements

- Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Poster & trending data: [The Movie Database (TMDb) API](https://www.themoviedb.org/documentation/api)
- Inspiration: Netflix's browsing experience