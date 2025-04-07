import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("movies_data.csv")

# Attempt to load the pickle file and ensure tensors are moved to CPU if necessary
embeddings_dict = None

try:
    # Open the pickle file and load the embeddings dictionary
    with open('embeddings.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)

    print("Pickle file loaded successfully.")
    
    # Check if embeddings_dict is None or empty
    if not embeddings_dict:
        print("embeddings_dict is empty or None.")
    
    # Check the type of each entry in embeddings_dict
    for key, value in embeddings_dict.items():
        print(f"Key: {key}, Type: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"Tensor shape for {key}: {value.shape}")
    
    # Move each tensor in the embeddings_dict to CPU if it's on CUDA
    for key in embeddings_dict:
        if isinstance(embeddings_dict[key], torch.Tensor):
            embeddings_dict[key] = embeddings_dict[key].to(torch.device('cpu'))  # Move tensor to CPU

except Exception as e:
    print(f"Error loading pickle file: {e}")

# If embeddings_dict is loaded successfully, access individual embeddings
if embeddings_dict is not None:
    genre_embeddings = embeddings_dict.get('genre', None)
    description_embeddings = embeddings_dict.get('description', None)
    cast_embeddings = embeddings_dict.get('cast', None)
    title_embeddings = embeddings_dict.get('title', None)
    director_embeddings = embeddings_dict.get('director', None)

    if genre_embeddings is None:
        print("Key 'genre' not found in embeddings_dict")
    if description_embeddings is None:
        print("Key 'description' not found in embeddings_dict")

    # Weighting factors
    genre_weight = 0.4
    description_weight = 0.4
    title_weight = 0.1
    cast_weight = 0.05
    director_weight = 0.05

    # Combine embeddings by weighted sum
    combined_embeddings = (
        genre_embeddings * genre_weight +
        description_embeddings * description_weight +
        cast_embeddings * cast_weight +
        title_embeddings * title_weight +
        director_embeddings * director_weight
    )

    # Continue with the recommendation function...
    print("Embeddings loaded and combined successfully.")
else:
    print("Embeddings could not be loaded successfully.")


def recommend_similar_movies_by_name(movie_name, df, combined_embeddings):
    # Find the index of the movie based on the title
    movie_index = df[df['title'] == movie_name].index[0]
    
    # Print the details of the movie that the user gives
    movie = df.iloc[movie_index]
    
    st.write(f"**Details of the movie '{movie_name}':**")
    st.write(f"**Title:** {movie['title']}")
    st.write(f"**Director:** {movie['director']}")
    st.write(f"**Cast:** {movie['cast']}")
    st.write(f"**Genres:** {movie['listed_in']}")
    st.write(f"**Rating:** {movie['rating']}")
    st.write(f"**Description:** {movie['description']}")
    st.write("\n---\n")
    
    # Calculate the cosine similarity matrix between all combined movie embeddings
    similarity_matrix = cosine_similarity(combined_embeddings.cpu().detach().numpy())

    # Find the most similar movies
    similar_movie_indices = similarity_matrix[movie_index].argsort()[-6:-1][::-1]  # Sort in descending order

    st.write(f"**Movies similar to '{movie_name}':**")
    
    for idx in similar_movie_indices:
        movie = df.iloc[idx]
        similarity_percentage = similarity_matrix[movie_index][idx] * 100  # Convert similarity to percentage
        
        # Print the movie details and similarity percentage
        st.write(f"**Title:** **{movie['title']}**")
        st.write(f"**Director:** {movie['director']}")
        st.write(f"**Cast:** {movie['cast']}")
        st.write(f"**Genres:** {movie['listed_in']}")
        st.write(f"**Rating:** {movie['rating']}")
        st.write(f"**Description:** {movie['description']}")
        st.write(f"**Similarity:** {similarity_percentage:.2f}%")
        st.write("\n---\n")

# Streamlit UI setup
st.title("Movie Recommendation System")
st.write("Select a movie from the dropdown to get recommendations based on your choice!")

# Create a dropdown menu to select the movie
movie_name = st.selectbox("Choose a movie:", [''] + df['title'].tolist())

# Once the user selects a movie, show recommendations
if movie_name:
    recommend_similar_movies_by_name(movie_name, df, combined_embeddings)
