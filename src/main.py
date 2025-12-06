import os
import sys
import streamlit as st
from recommend import df, recommend_songs

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    #page_layout="centered"
)

st.title("🎶 Instant Music Recommender")

song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

if st.button("🎶 Recommend"):
    with st.spinner("🔍 Finding similar songs..."):
        recommendations = recommend_songs(selected_song)
        if recommendations is None:
            st.warning("Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)


