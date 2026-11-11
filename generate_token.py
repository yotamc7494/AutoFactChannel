import base64
import pickle
import json
from google_auth_oauthlib.flow import InstalledAppFlow

# אותם הסקופים שהגדרת בקוד ההעלאה
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.force-ssl',
]


def generate_base64_token_from_scratch(client_secrets_file="client_secret.json"):
    """
    מבצע אימות OAuth מול גוגל ומדפיס את ה-Base64 Token
    ללא שמירת קובץ pickle מקומי.
    """
    print("🚀 מתחיל תהליך אימות מול Google...")

    # 1. יצירת ה-Flow מתוך קובץ ה-JSON של הגוגל קלאוד
    flow = InstalledAppFlow.from_client_secrets_file(
        client_secrets_file,
        scopes=SCOPES
    )

    # 2. הרצת השרת המקומי לאימות בדפדפן (יפתח חלון בדפדפן להתחברות)
    creds = flow.run_local_server(port=0)

    # 3. המרת ה-Credentials ל-Bytes בעזרת pickle בזיכרון בלבד
    pickle_bytes = pickle.dumps(creds)

    # 4. המרה ל-Base64 string
    base64_str = base64.b64encode(pickle_bytes).decode("utf-8")

    print("\n✅ האימות הושלם בהצלחה!")
    print("=" * 60)
    print("העתק את המחרוזת הבאה ל-st.secrets תחת המפתח YOUTUBE_TOKEN_PICKLE_BASE64:\n")
    print(base64_str)
    print("=" * 60)


if __name__ == "__main__":
    # ודא שקובץ ה-OAuth Client Secret שהורדת מ-Google Cloud
    # נמצא באותה תיקייה (או עדכן את השם בפרמטר)
    generate_base64_token_from_scratch("client_secret.json")