import os
import streamlit as st
from main import run_shorts_pipeline

st.set_page_config(page_title="YouTube Shorts Generator", page_icon="🎬", layout="centered")

st.title("🎬 מחולל סרטוני YouTube Shorts")
st.write("לחץ על אחד הכפתורים כדי להפעיל את פייפליין הייצור:")

col1, col2 = st.columns(2)

video_path = None

with col1:
    if st.button("🚀 צור והעלה ל-YouTube", use_container_width=True):
        with st.spinner("מייצר תסריט, תמונות, שמע ומעלה ל-YouTube..."):
            try:
                video_path = run_shorts_pipeline(upload=True)
                st.success("✅ הסרטון נוצר והועלה בהצלחה!")
            except Exception as e:
                st.error(f"🚨 אירעה שגיאה: {e}")

with col2:
    if st.button("🎬 צור בלבד (ללא העלאה)", use_container_width=True):
        with st.spinner("מייצר תסריט, תמונות ושמע..."):
            try:
                video_path = run_shorts_pipeline(upload=False)
                st.success("✅ הסרטון נוצר בהצלחה!")
            except Exception as e:
                st.error(f"🚨 אירעה שגיאה: {e}")

# הצגת הווידאו המוכן בסיום ההרצה
if video_path and os.path.exists(video_path):
    st.divider()
    st.subheader("📹 צפייה בסרטון המוכן:")
    st.video(video_path)