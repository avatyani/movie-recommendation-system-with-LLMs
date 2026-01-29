import streamlit as st
import openai
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests
import json
import os
from openai import OpenAI

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        st.error("OpenAI API key not found! Please add OPENAI_API_KEY to your Streamlit Cloud secrets.")
        return None
    return OpenAI(api_key=api_key)

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
    api_key = "99445886"
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=5).json()
        return response.get('Poster', None)
    except:
        return None

fallback_url = "https://st4.depositphotos.com/14953852/24787/v/450/depositphotos_247872612-stock-illustration-no-image-available-icon-vector.jpg"

# Load our custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>Movie Recommendation System</h1>", unsafe_allow_html=True)

# Only show dropdown message on home page
if st.session_state.page == "home":
    st.write("Select a movie from the dropdown to get AI-powered recommendations!")

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

# Go back button
def go_back():
    st.session_state.page = 'home'

# Detail page
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
                    f"<div class='similarity'><strong>🎯 {score:.2f}% Match</strong></div>",
                    unsafe_allow_html=True
                )
            except:
                st.warning("Could not calculate similarity.")

    st.write("")
    st.button("⬅️ Go Back", on_click=go_back)


# Stage 1: Get embedding-based candidates
def get_embedding_candidates(movie_name, df, combined_embeddings, k=10):
    """
    Get top-k candidates using embedding similarity (fast retrieval)
    """
    matches = df[df['title'] == movie_name]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    
    # Calculate similarities
    sims = cosine_similarity(combined_embeddings.cpu().detach().numpy())
    similar_idxs = sims[idx].argsort()[-k-1:-1][::-1]
    
    candidates = []
    for i in similar_idxs:
        m = df.iloc[i]
        candidates.append({
            'title': m['title'],
            'similarity': float(sims[idx][i]),
            'genre': m['listed_in'],
            'description': m['description'],
            'director': m.get('director', 'Unknown'),
            'cast': m.get('cast', 'Unknown'),
            'rating': m.get('rating', 'N/A')
        })
    
    return candidates


# Stage 2: LLM re-ranking and explanation
@st.cache_data(show_spinner=False, ttl=3600)
def llm_rerank_and_explain(selected_movie: str, selected_genre: str, selected_description: str, 
                           candidates_json: str):
    """
    Use OpenAI GPT to re-rank candidates and generate explanations
    Cache results for 1 hour to reduce API costs
    """
    client = get_openai_client()
    if not client:
        return None
    
    candidates = json.loads(candidates_json)
    
    # Build candidates text
    candidates_text = ""
    for i, c in enumerate(candidates):
        candidates_text += f"{i+1}. {c['title']} (Embedding Similarity: {c['similarity']:.2%})\n"
        candidates_text += f"   Genre: {c['genre']}\n"
        candidates_text += f"   Description: {c['description'][:200]}...\n\n"
    
    prompt = f"""You are an expert movie recommendation analyst. Analyze movie similarities based on themes, storytelling style, tone, and narrative elements.

USER SELECTED MOVIE:
Title: {selected_movie}
Genre: {selected_genre}
Description: {selected_description}

CANDIDATE RECOMMENDATIONS (from embedding similarity):
{candidates_text}

TASK:
Select the TOP 5 most relevant recommendations from these candidates. For each:
1. Explain WHY it's similar (focus on specific themes, tone, storytelling style, character dynamics)
2. Rate the relevance from 1-10

Return ONLY a JSON array with this exact format:
[
  {{
    "title": "Movie Title",
    "explanation": "2-3 sentences explaining specific similarities in themes, tone, or style",
    "relevance": 9
  }}
]

Focus on meaningful similarities, not just genre matching. Be specific about what makes each movie similar."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cost-effective
            messages=[
                {"role": "system", "content": "You are a movie recommendation expert who provides insightful, specific analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"LLM API Error: {e}")
        return None


def parse_llm_response(llm_response):
    """
    Parse LLM JSON response safely
    """
    if not llm_response:
        return None
    
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
            return recommendations
        else:
            st.error("Could not parse LLM response")
            return None
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        return None


# Main hybrid recommendation function
def hybrid_recommend_similar_movies(movie_name, df, combined_embeddings):
    """
    Hybrid recommendation: Embeddings (fast) + LLM (quality)
    """
    matches = df[df['title'] == movie_name]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    movie = df.iloc[idx]
    
    # Display selected movie info
    st.markdown(f"<span class='movie-title'>**Selected:** {movie['title']}</span>", unsafe_allow_html=True)
    st.write(f"**Director:** {movie['director']}")
    st.write(f"**Genres:** {movie['listed_in']}")
    st.write(f"**Description:** {movie['description']}")
    st.write("---")
    
    # Stage 1: Fast embedding-based retrieval
    with st.spinner("🔍 Finding similar movies using embeddings..."):
        candidates = get_embedding_candidates(movie_name, df, combined_embeddings, k=10)
    
    if not candidates:
        st.warning("No similar movies found.")
        return
    
    # Stage 2: LLM re-ranking and explanation
    with st.spinner("🤖 Analyzing recommendations with AI (GPT-4)..."):
        candidates_json = json.dumps(candidates)
        llm_response = llm_rerank_and_explain(
            movie_name, 
            movie['listed_in'],
            movie['description'],
            candidates_json
        )
        
        recommendations = parse_llm_response(llm_response)
    
    if not recommendations:
        st.warning("LLM analysis failed. Showing embedding-based results:")
        # Fallback to embedding results
        recommendations = [
            {
                'title': c['title'],
                'explanation': f"Similar based on genre and plot elements. Embedding similarity: {c['similarity']:.1%}",
                'relevance': int(c['similarity'] * 10)
            }
            for c in candidates[:5]
        ]
    
    # Display recommendations
    st.markdown("### 🎬 AI-Powered Recommendations")
    st.caption("*Ranked and explained by GPT-4 based on themes, tone, and storytelling style*")
    
    for rec in recommendations:
        title = rec['title']
        explanation = rec.get('explanation', 'No explanation available')
        relevance = rec.get('relevance', 0)
        
        # Get movie details
        movie_data = df[df['title'] == title]
        if movie_data.empty:
            continue
        
        m = movie_data.iloc[0]
        
        # Get poster
        poster_url = get_poster_url(m['title'])
        if not poster_url or poster_url == "N/A":
            poster_url = fallback_url
        
        # Get embedding similarity for reference
        m_idx = movie_data.index[0]
        sims = cosine_similarity(combined_embeddings.cpu().detach().numpy())
        embedding_score = sims[idx][m_idx] * 100
        
        # Show the card with LLM explanation
        st.markdown(f"""
        <div class="recommendation-card">
            <div style="display: flex; gap: 1rem;">
                <img src="{poster_url}" width="100" style="border-radius: 8px;" />
                <div style="flex: 1;">
                    <div class="movie-title">{m['title']}</div>
                    <div style="margin: 0.5rem 0;">
                        <span style="color: #90ee90; font-weight: bold;">🤖 AI Relevance: {relevance}/10</span>
                        <span style="margin-left: 1rem; color: #888;">📊 Embedding Score: {embedding_score:.1f}%</span>
                    </div>
                    <div style="margin: 0.5rem 0; font-style: italic; color: #ccc;">
                        "{explanation}"
                    </div>
                    <div style="margin-top: 0.5rem; color: #888; font-size: 0.9rem;">
                        <strong>Genre:</strong> {m['listed_in']} | <strong>Rating:</strong> {m.get('rating', 'N/A')}
                    </div>
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

    movie_name = st.selectbox("", titles, index=selected_index)

    if movie_name and movie_name != placeholder_option:
        st.session_state.selected_input_movie = movie_name

    # Always show recommendations if a valid movie was selected
    if st.session_state.selected_input_movie and st.session_state.selected_input_movie != placeholder_option:
        hybrid_recommend_similar_movies(st.session_state.selected_input_movie, df, combined_embeddings)

elif st.session_state.page == "details":
    show_movie_details(df, st.session_state.selected_movie)
