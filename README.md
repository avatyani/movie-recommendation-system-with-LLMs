# 🎬 Movie Recommendation System with LLMs

This is a personalized movie recommendation system powered by **Sentence-BERT** (a transformer-based large language model) and **PyTorch**, built to provide intelligent suggestions based on semantic similarity between movie descriptions.

## 🚀 Features

- 🔍 **Semantic Recommendations**: Uses Sentence-BERT to compare plot descriptions and find similar movies.
- 🎨 **Modern UI**: Built with Streamlit, featuring a dark theme and clean, responsive design.
- 🎥 **Movie Details Page**: View metadata like genre, type, cast, release year, duration, and overview.
- 📊 **Filters**: Filter recommendations by genre, type, release year, and duration.
- 🖼️ **Image Handling**: Placeholder images for missing posters.
- ⚙️ **Backend Efficiency**: Utilizes cosine similarity and precomputed embeddings for real-time performance.

## 🧠 Tech Stack

- **LLM**: Sentence-BERT for semantic embedding
- **Framework**: PyTorch, Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit
- **Deployment**: Ready for local and cloud deployment

## 🌐 Live Demo

https://movie-discovery.streamlit.app/

## 🧪 How It Works

1. Preprocess and embed movie overviews using Sentence-BERT.
2. Use cosine similarity to compute nearest neighbors.
3. Display top-k similar movies via a dropdown.
4. Show movie metadata and recommendations in a user-friendly UI.

## 📁 Project Structure
📦movie-recommendation-system-with-LLMs ┣ 📁 data/ ┃ ┗ movies_metadata.csv ┣ 📁 model/ ┃ ┗ embeddings.pkl ┣ 📄 app.py ┣ 📄 utils.py ┣ 📄 requirements.txt ┗ 📄 README.md

## 🖥️ Getting Started

```bash
git clone https://github.com/avatyani/movie-recommendation-system-with-LLMs.git
cd movie-recommendation-system-with-LLMs
pip install -r requirements.txt
streamlit run app.py

