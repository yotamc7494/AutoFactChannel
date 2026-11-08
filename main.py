from dotenv import load_dotenv
load_dotenv()
from gradio_client import Client

# חיבור ל-Space החינמי של FLUX.1-schnell
client = Client("black-forest-labs/FLUX.1-schnell")

result = client.predict(
    prompt="A hyper-realistic close up of an ancient Roman coin, cinematic lighting, 8k",
    seed=0,
    randomize_seed=True,
    width=720,
    height=1280,
    num_inference_steps=4,
    api_name="/infer"
)

# result מחזיר נתיב לקובץ התמונה הזמני שנשמר במחשב
print(f"Success! Image saved to: {result[0]}")