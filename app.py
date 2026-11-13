import os
import streamlit as st
from main import run_shorts_pipeline
from streamlit.logger import get_logger
st.set_page_config(page_title="YouTube Shorts Generator", page_icon="🎬", layout="centered")

# --- 1. טיפול בקריאות Webhook מרחוק (Ping / Upload) ---
WEBHOOK_SECRET = st.secrets.get("WEBHOOK_SECRET")
query_params = st.query_params

trigger = query_params.get("trigger")
provided_secret = query_params.get("secret")

logger = get_logger(__name__)

if trigger:
    if provided_secret == WEBHOOK_SECRET:
        # PING קל: מונע הירדמות ללא הפעלת הפייפליין הכבד
        if trigger == "ping":
            print("ping")
            logger.warning("🟢 [PING] Keep-Alive Ping Received!")
            st.write("PONG 🏓 - App is awake and ready!")
            st.stop()

        # UPLOAD: הפעלת פייפליין הייצור והתזמון המלא
        elif trigger == "upload":
            st.info("⚡ התקבלה קריאת Webhook להעלאה. מפעיל פייפליין...")
            try:
                video_path = run_shorts_pipeline(upload=True)
                st.success("✅ הפייפליין הושלם והסרטון תומזן/הועלה בהצלחה!")
                if video_path and os.path.exists(video_path):
                    st.video(video_path)
            except Exception as e:
                st.error(f"🚨 שגיאה בביצוע הפייפליין: {e}")
            st.stop()
    else:
        st.error("🚨 קריאת Webhook נכשלה: מפתח Secret לא תקין.")
        st.stop()

# --- 2. ממשק ידני (כפתורים בדפדפן) ---
st.title("🎬 מחולל סרטוני YouTube Shorts")
st.write("לחץ על אחד הכפתורים להפעלה ידנית:")

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

if video_path and os.path.exists(video_path):
    st.divider()
    st.subheader("📹 צפייה בסרטון המוכן:")
    st.video(video_path)