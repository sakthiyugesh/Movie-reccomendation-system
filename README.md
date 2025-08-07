# 🎬 Movie Recommendation System

A simple yet effective **Movie Recommendation System** built using **TF-IDF Vectorization** and **Cosine Similarity**. It suggests similar movies based on the movie title selected by the user.

🔗 **Live App:** [https://sakthi.streamlit.app](https://sakthi.streamlit.app)

---

## 🚀 Features

- Suggests top similar movies based on a selected movie title.
- Uses **TF-IDF vectorizer** to transform movie overviews into numerical representations.
- Computes **cosine similarity** to identify closely related movies.
- Deployed and accessible online via **Streamlit Cloud**.

---

## 🧠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Libraries:** 
  - `pandas`
  - `sklearn` (TF-IDF Vectorizer, cosine_similarity)
  - `streamlit`

---

## 📦 How It Works

1. **TF-IDF Vectorization**: Transforms textual overviews of movies into vectors.
2. **Cosine Similarity**: Measures the similarity between movies based on vector distance.
3. **User Interface**: User selects a movie from dropdown → System shows top similar movies.

---


## 🛠️ How to Run Locally

```bash
git clone https://github.com/sakthiyugesh/movie-recommendation-system
cd movie-recommendation-system
pip install -r requirements.txt
streamlit run app.py
