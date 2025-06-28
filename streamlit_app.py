from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from time import sleep
import streamlit as st
from difflib import get_close_matches

# --- Session State ---
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = ""
if "visible_count" not in st.session_state:
    st.session_state.visible_count = 6

# --- Load and preprocess data ---
data = pd.read_csv('top10K-TMDB-movies.csv')
df = pd.DataFrame(data)
new_data = df.dropna()
new_data['final_tags'] = new_data['genre'] + new_data['overview']
final_data = new_data.drop(['genre', 'original_language', 'overview', 'popularity',
                            'vote_average', 'vote_count', 'release_date'], axis=1)

# --- TF-IDF + KNN Setup ---
tfvectorize = TfidfVectorizer(stop_words='english')
tf_matrix = tfvectorize.fit_transform(final_data['final_tags'])
model = NearestNeighbors(n_neighbors=16, metric='cosine')
model.fit(tf_matrix)
indices = pd.Series(final_data.index, index=final_data['title'])

# --- Recommendation Function ---
def reccommend_movie_KN(title, n):
    idx = indices[title]
    _, neighbors = model.kneighbors(tf_matrix[idx], n_neighbors=n + 1)
    return list(final_data['title'].iloc[neighbors[0][1:n+1]])

# --- TMDb API Call ---
def get_movies(title):
    API_KEY = 'a01b8453d45c3f5806c7397aa4addae5'
    BASE_URL = 'https://api.themoviedb.org/3/search/movie'
    params = {'api_key': API_KEY, 'query': title}
    res = requests.get(BASE_URL, params=params)
    data = res.json()
    if data['results']:
        movie = data['results'][0]
        return {
            'title': movie['title'],
            'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
            'rating': movie['vote_average'],
            'release_date': movie['release_date'],
            'overview': movie.get('overview', '')
        }
    return {'title': title, 'error': 'Not found'}

# --- Page Config + Styling ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .suggestion-buttons {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
            
    }
    .movie-text {
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 6;
        -webkit-box-orient: vertical;
        font-size: 0.9rem;
    }
    .skeleton {
        background-color: #333;
        border-radius: 8px;
        height: 300px;
        width: 100%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# --- Header (Centered Title + Search Box) ---
with st.container():
    st.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center; text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 0.2rem;'>🎬 Movie Recommender</h1>
            <p style='color: gray; margin-top: 0;'>Find similar movies based on your favorites</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 20, 1])
    with col2:
        query = st.text_input("Search a movie title:", placeholder="Type here").strip()

# --- Suggestions ---
all_titles = list(final_data['title'])
matches = get_close_matches(query, all_titles, n=5, cutoff=0.4) if query else []

if query in all_titles:
    st.session_state.selected_movie = query
    st.session_state.visible_count = 6

if query and matches:
    
    # Create centered buttons using columns
    cols = st.columns(len(matches))
    for i, match in enumerate(matches):
        with cols[i]:
            if st.button(
                match, 
                key=f"suggest-{i}",
                # Optional styling:
                use_container_width=True,
                help=f"Search for {match}"
            ):
                st.session_state.selected_movie = match
                st.session_state.visible_count = 6
                st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)


# --- Recommendations Display ---
if st.session_state.selected_movie:
    selected_title = st.session_state.selected_movie
    st.subheader(f"Recommendations for: {selected_title}")
    movie_list = reccommend_movie_KN(selected_title, 15)

    total_movies_list = []
    with st.spinner("Fetching movie details..."):
        for m in movie_list:
            movie = get_movies(m)
            total_movies_list.append(movie)
            sleep(0.1)

    visible_movies = total_movies_list[:st.session_state.visible_count]

    for i in range(0, len(visible_movies), 6):
        cols = st.columns(6)
        for j in range(6):
            if i + j < len(visible_movies):
                movie = visible_movies[i + j]
                with cols[j]:
                    if movie.get("poster_url"):
                        st.image(movie["poster_url"], use_container_width=True)
                    else:
                        st.markdown("<div class='skeleton'></div>", unsafe_allow_html=True)
                    st.markdown(f"**{movie['title']}**", unsafe_allow_html=True)
                    st.caption(f"⭐ {movie['rating']} | 📅 {movie['release_date']}")
                    st.markdown(f"<p class='movie-text'>{movie['overview']}</p>", unsafe_allow_html=True)

    # --- Show More Button ---
    if st.session_state.visible_count < len(total_movies_list):
        if st.button("Show More"):
            st.session_state.visible_count += 6
            st.rerun()
else:
    if query:
        st.error("Movie not found. Please try a different title or pick a suggestion.")
    else:
        st.info("Type a movie name to get recommendations.")
