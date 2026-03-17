import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------
# STREAMLIT TITLE
# ------------------------------
st.title("Spotify Songs Genre Segmentation & Recommendation System")

# ------------------------------
# LOAD DATASET
# ------------------------------
df = pd.read_csv("spotify_songs.csv.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# ------------------------------
# AUDIO FEATURES
# ------------------------------
audio_features = [
'danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness','valence','tempo'
]

# ------------------------------
# GENRE DISTRIBUTION
# ------------------------------
st.subheader("Genre Distribution")

fig1 = plt.figure(figsize=(10,5))
sns.countplot(data=df, x='playlist_genre', order=df['playlist_genre'].value_counts().index)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ------------------------------
# CORRELATION MATRIX
# ------------------------------
st.subheader("Correlation Matrix")

corr_features = audio_features + ['track_popularity','duration_ms']
corr_matrix = df[corr_features].corr()

fig2 = plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, cmap="coolwarm")
st.pyplot(fig2)

# ------------------------------
# DATA SCALING
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[audio_features])

# ------------------------------
# KMEANS CLUSTERING
# ------------------------------
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# ------------------------------
# PCA VISUALIZATION
# ------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]

st.subheader("K-Means Clusters Visualization")

fig3 = plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set1')
st.pyplot(fig3)

# ------------------------------
# RECOMMENDATION FUNCTION
# ------------------------------
def recommend_songs(track_id, dataframe, num_recommendations=5):

    try:
        target_cluster = dataframe[dataframe['track_id']==track_id]['cluster'].iloc[0]
    except:
        return "Track ID not found"

    recommendations = dataframe[
        (dataframe['cluster']==target_cluster) &
        (dataframe['track_id']!=track_id)
    ]

    recommendations = recommendations.sort_values(
        by='track_popularity',
        ascending=False
    )

    return recommendations[
        ['track_name','track_artist','playlist_genre','track_popularity']
    ].head(num_recommendations)

# ------------------------------
# SONG SELECTION
# ------------------------------
st.subheader("Song Recommendation")

song_list = df['track_name'].unique()

selected_song = st.selectbox("Select a Song", song_list)

if st.button("Recommend Songs"):

    track_id = df[df['track_name']==selected_song].iloc[0]['track_id']

    recommendations = recommend_songs(track_id, df)

    st.write("Recommended Songs:")
    st.dataframe(recommendations)

# ------------------------------
# FOOTER
# ------------------------------
st.write("Project completed successfully.")