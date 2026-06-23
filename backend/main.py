# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "¡FastAPI levantado a la velocidad de la luz! 🚀"}