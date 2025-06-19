import streamlit as st
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'selected_input_movie' not in st.session_state:
    st.session_state.selected_input_movie = None
if st.session_state.get("_rerun", False):
    st.session_state._rerun = False
    st.rerun()

# Fetching movie poster from omdb with api key
@st.cache_data(show_spinner=False)
def get_poster_url(title):
    api_key = "99445886"  # <-- replace this with your OMDb API key
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url).json()
    return response.get('Poster', None)
fallback_url = "https://st4.depositphotos.com/14953852/24787/v/450/depositphotos_247872612-stock-illustration-no-image-available-icon-vector.jpg"


# Load our custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>Movie Recommendation System</h1>", unsafe_allow_html=True)
# Only show dropdown message on home page
if st.session_state.page == "home":
    st.write("Select a movie from the dropdown to get recommendations based on your choice!")

# Load data
df = pd.read_csv("movies_data.csv")

# Load embeddings
embeddings_dict = None
try:
    with open('embeddings.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
    # Move any CUDA tensors to CPU
    for k, v in embeddings_dict.items():
        if isinstance(v, torch.Tensor):
            embeddings_dict[k] = v.to(torch.device('cpu'))
except Exception as e:
    st.error(f"Error loading embeddings: {e}")

# Combine embeddings if available
if embeddings_dict:
    genre_emb = embeddings_dict.get('genre')
    desc_emb = embeddings_dict.get('description')
    cast_emb = embeddings_dict.get('cast')
    title_emb = embeddings_dict.get('title')
    dir_emb = embeddings_dict.get('director')

    combined_embeddings = (
        genre_emb * 0.4 +
        desc_emb * 0.4 +
        cast_emb * 0.05 +
        title_emb * 0.1 +
        dir_emb * 0.05
    )
else:
    combined_embeddings = None

# go back button
def go_back():
    st.session_state.page = 'home'
    # st.session_state._rerun = True  # set a custom flag

# detail page
def show_movie_details(df, title):
    m = df[df['title'] == title].iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        poster_url = get_poster_url(m['title'])
        if not poster_url or poster_url == "N/A":
            poster_url = fallback_url
        if poster_url and poster_url != "N/A":
            st.image(poster_url, width=200)
    with col2:
        st.markdown(f"<h2 class='movie-title'>{m['title']}</h2>", unsafe_allow_html=True)
        st.write(f"**Type:** {m.get('type', 'N/A')}")
        st.write(f"**Release Year:** {m.get('release_year', 'N/A')}")

        duration_raw = m.get('duration', 'N/A')
        if isinstance(duration_raw, str):
            if "min" in duration_raw:
                st.write(f"**Duration:** {duration_raw.strip()}")
            elif "Season" in duration_raw:
                st.write(f"**Seasons:** {duration_raw.strip()}")
            else:
                st.write(f"**Duration Info:** {duration_raw.strip()}")
        else:
            st.write(f"**Duration:** N/A")

        st.write(f"**Director:** {m.get('director', 'N/A')}")
        cast_full = m.get('cast', '')
        if cast_full:
            cast_list = [c.strip() for c in cast_full.split(',')]
            top_cast = ', '.join(cast_list[:2])
            st.write(f"**Cast:** {top_cast}...")

            if len(cast_list) > 2:
                with st.expander("🎭 See full cast"):
                    st.write(', '.join(cast_list))
        else:
            st.write("**Cast:** N/A")

        st.write(f"**Genres:** {m.get('listed_in', 'N/A')}")
        st.write(f"**Rating:** {m.get('rating', 'N/A')}")
        st.write(f"**Description:** {m.get('description', 'N/A')}")

        # Similarity badge
        if (
            st.session_state.selected_input_movie
            and st.session_state.selected_input_movie != title
            and combined_embeddings is not None
        ):
            try:
                i1 = df[df['title'] == st.session_state.selected_input_movie].index[0]
                i2 = df[df['title'] == title].index[0]
                score = cosine_similarity(
                    combined_embeddings[i1].reshape(1, -1),
                    combined_embeddings[i2].reshape(1, -1)
                )[0][0] * 100

                st.markdown(
                    f"<div class='similarity'><strong> {score:.2f}% Match</strong></div>",
                    unsafe_allow_html=True
                )
            except:
                st.warning("Could not calculate similarity.")

    st.write("")
    st.button("⬅️ Go Back", on_click=go_back)



# Recommend movies
def recommend_similar_movies(movie_name, df, combined_embeddings):
    matches = df[df['title'] == movie_name]
    if matches.empty:
        return []
    idx = matches.index[0]
    movie = df.iloc[idx]

    st.markdown(f"<span class='movie-title'>**Title:** {movie['title']}</span>", unsafe_allow_html=True)
    st.write(f"**Director:** {movie['director']}")
    st.write(f"**Cast:** {movie['cast']}")
    st.write(f"**Genres:** {movie['listed_in']}")
    st.write(f"**Rating:** {movie['rating']}")
    st.write(f"**Description:** {movie['description']}")
    st.write("---")

    # Similarities
    sims = cosine_similarity(combined_embeddings.cpu().detach().numpy())
    similar_idxs = sims[idx].argsort()[-6:-1][::-1]

    st.write(f"**Movies similar to '{movie_name}':**")

    for i in similar_idxs:
        m = df.iloc[i]
        poster_url = get_poster_url(m['title'])
        if not poster_url or poster_url == "N/A":
            poster_url = fallback_url

        description_preview = m['description'][:150] + "..." if len(m['description']) > 150 else m['description']
        score = sims[idx][i] * 100

        # Show the card
        st.markdown(f"""
        <div class="recommendation-card">
            <div style="display: flex; gap: 1rem;">
                <img src="{poster_url}" width="100" style="border-radius: 8px;" />
                <div>
                    <div class="movie-title">{m['title']}</div>
                    <div style="margin-top: 0.5rem;">{description_preview}</div>
                    <div class="similarity"><strong>{score:.2f}% Match</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show a regular button below the card
        if st.button(f"View Details - {m['title']}", key=f"btn-{m['title']}"):
            st.session_state.page = "details"
            st.session_state.selected_movie = m['title']
            st.rerun()


# Home page logic
if st.session_state.page == "home":
    st.markdown("<label class='choose-movie'>Choose a movie:</label>", unsafe_allow_html=True)

    placeholder_option = "🔍 Search... "
    titles = [placeholder_option] + df['title'].tolist()
    selected_index = titles.index(st.session_state.selected_input_movie) if st.session_state.selected_input_movie in titles else 0

    movie_name = st.selectbox("", titles)

    if movie_name and movie_name != placeholder_option:
        st.session_state.selected_input_movie = movie_name

    # always show recommendations if a valid movie was selected earlier
    if st.session_state.selected_input_movie and st.session_state.selected_input_movie != placeholder_option:
        recommend_similar_movies(st.session_state.selected_input_movie, df, combined_embeddings)



elif st.session_state.page == "details":
    show_movie_details(df, st.session_state.selected_movie)