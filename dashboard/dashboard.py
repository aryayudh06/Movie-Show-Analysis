import ast
import streamlit as st
from pymongo import MongoClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Movie Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def get_data_from_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["OLAP"]
        collection = db["media"]
        data = list(collection.find())
        if not data:
            st.warning("Tidak ada data yang ditemukan di koleksi MovieShow")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df.drop(columns=['_id'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data dari MongoDB: {str(e)}")
        return pd.DataFrame()
    finally:
        if 'client' in locals():
            client.close()

def preprocess_data(df):
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year

    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
    
    return df

def sidebar_filters(df):
    st.sidebar.header("Filter")

    min_rating = st.sidebar.slider("Rating Minimum", float(df['rating'].min()), 10.0, float(df['rating'].min()), 0.1)

    all_genres = set()
    for genre_list in df['genres'].dropna():
        if isinstance(genre_list, list):
            all_genres.update([g.strip() for g in genre_list if isinstance(g, str)])
        
    genre_options = sorted(all_genres)
    selected_genres = st.sidebar.multiselect("Genre", options=genre_options)

    if 'release_year' in df.columns:
        min_year = int(df['release_year'].min())
        max_year = int(df['release_year'].max())
        selected_year_range = st.sidebar.slider("Tahun Rilis", min_year, max_year, (min_year, max_year))
    else:
        selected_year_range = None

    return min_rating, selected_genres, selected_year_range

def filter_data(df, min_rating, selected_genres, selected_year_range):
    df_filtered = df[df['rating'] >= min_rating].copy()

    if selected_genres:
        df_filtered = df_filtered[df_filtered['genres'].apply(lambda g: any(s in g for s in selected_genres))]

    if selected_year_range and 'release_year' in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered['release_year'].between(selected_year_range[0], selected_year_range[1])
        ]

    return df_filtered

# -----Main App-----
df = get_data_from_mongodb()
sns.set_theme(style="whitegrid", palette="crest")

if not df.empty:
    df = preprocess_data(df)
    min_rating, selected_genres, selected_year_range = sidebar_filters(df)
    df_filtered = filter_data(df, min_rating, selected_genres, selected_year_range)
    df_genres_exploded = df_filtered.explode('genres')
    df_genres_exploded = df_genres_exploded[df_genres_exploded['genres'].notna() & (df_genres_exploded['genres'] != '')]

    st.title("Movie Analytics Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_film = len(df_filtered)
        avg_rating = df_filtered['rating'].mean() if not df_filtered.empty else 0
        min_year = df_filtered['release_year'].min() if 'release_year' in df_filtered.columns else '-'
        max_year = df_filtered['release_year'].max() if 'release_year' in df_filtered.columns else '-'
        col4, col5 = st.columns(2)
        with col4:
            st.metric(label="Total Film", value=f"{total_film}")
        with col5:
            st.metric(label="Rata-rata Rating", value=f"{avg_rating:.2f}")
        col6, col7=st.columns(2)
        with col6:
            st.metric(label="Tahun Rilis Pertama", value=f"{min_year:.0f}")
        with col7:
            st.metric(label="Tahun Rilis Terbaru", value=f"{max_year:.0f}")

        if 'release_year' in df_filtered.columns:
            film_per_year = df_filtered['release_year'].value_counts().sort_index()
            fig1, ax1 = plt.subplots(figsize=(12, 5.5))
            sns.lineplot(x=film_per_year.index, y=film_per_year.values, color="#2E5E86", ax=ax1)
            ax1.set_title("Jumlah Film Dirilis per Tahun", fontsize=16)
            ax1.set_xlabel("Tahun", fontsize=12)
            ax1.set_ylabel("Jumlah Film", fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
        else:
            st.write("Data 'release_year' tidak tersedia.")

        # Grafik rata-rata rating per tahun
        if 'release_year' in df_filtered.columns and 'rating' in df_filtered.columns:
            rating_per_year = df_filtered.groupby('release_year')['rating'].mean()
            fig2, ax2 = plt.subplots(figsize=(12, 5.5))
            sns.lineplot(x=rating_per_year.index, y=rating_per_year.values,color="#2E5E86", marker='o', ax=ax2)
            ax2.set_title("Rata-Rata Rating per Tahun", fontsize=16)
            ax2.set_xlabel("Tahun", fontsize=12)
            ax2.set_ylabel("Rating Rata-Rata", fontsize=12)
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.write("Data 'rating' atau 'release_year' tidak tersedia.")

    with col2:
        fig, ax = plt.subplots(figsize=(12, 11)) # Ukuran plot lebih besar
        sns.histplot(df_filtered['rating'], kde=True, ax=ax, bins=15)
        ax.set_title('Distribusi Rating Film', fontsize=16)
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Jumlah Film', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        st.pyplot(fig)

        top_genres = df_filtered.explode('genres')['genres'].value_counts().nlargest(5)
        if not top_genres.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax, palette="Blues_d")
            ax.set_title('Top 5 Genre Berdasarkan Jumlah', fontsize=16)
            ax.set_xlabel('Jumlah film', fontsize=12)
            ax.set_ylabel('Genre', fontsize=12)
            st.pyplot(fig, bbox_inches='tight', use_container_width=False)

    with col3:
        if 'rating' in df_filtered.columns and 'title' in df_filtered.columns:
            top_movies = df_filtered.sort_values(by='rating', ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(12, 4.5))
            sns.barplot(x='title', y='rating', data=top_movies, ax=ax, palette='crest')
            ax.set_title(f'Top 5 Film Berdasarkan Rating', fontsize=16)
            ax.set_xlabel('Rating', fontsize=12)
            ax.set_ylabel('Judul Film', fontsize=12)
            st.pyplot(fig, bbox_inches='tight',use_container_width=False)
            
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(x='type', data=df_filtered, palette='Blues_d', ax=ax)
        ax.set_title('Distribusi Tipe Film', fontsize=16)
        ax.set_xlabel('Tipe', fontsize=12)
        ax.set_ylabel('Jumlah Film', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        st.pyplot(fig)

        genre_rating = df_genres_exploded.groupby('genres')['rating'].mean().sort_values(ascending=False).head(5).reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='rating', y='genres', data=genre_rating, ax=ax, palette='crest')
        ax.set_title(f'Top 5 Genre Berdasarkan Rating', fontsize=16)
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        st.pyplot(fig, bbox_inches='tight',use_container_width=False)

        
