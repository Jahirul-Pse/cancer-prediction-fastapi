import webbrowser
import threading
import time
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from model import predict_majority, train_models
from fastapi.staticfiles import StaticFiles

# Load dataset
df = pd.read_csv("./dataset/The_Cancer_data_1500_V2.csv")  # ⬅️ Update this with your actual CSV filename

# Compute slider ranges dynamically from dataset
slider_ranges = {
    "Age": {"min": df["Age"].min(), "max": df["Age"].max()},
    "BMI": {"min": df["BMI"].min(), "max": df["BMI"].max()},
    "PhysicalActivity": {"min": df["PhysicalActivity"].min(), "max": df["PhysicalActivity"].max()},
    "AlcoholIntake": {"min": df["AlcoholIntake"].min(), "max": df["AlcoholIntake"].max()}
}
def round_to_step(value, step=0.1):
    return round(value / step) * step

# Round float slider ranges to nearest step for smooth slider behavior
for key in ["BMI", "PhysicalActivity", "AlcoholIntake"]:
    slider_ranges[key]["min"] = round_to_step(slider_ranges[key]["min"], 0.1)
    slider_ranges[key]["max"] = round_to_step(slider_ranges[key]["max"], 0.1)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000")

@app.on_event("startup")
def startup_event():
    train_models()  # Train once when app starts
    threading.Thread(target=open_browser).start()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "columns": columns,
        "prediction": None,
        "slider_ranges": slider_ranges
    })

@app.post("/", response_class=HTMLResponse)
async def post_form(
    request: Request,
    Age: int = Form(...),
    Gender: int = Form(...),
    BMI: float = Form(...),
    Smoking: int = Form(...),
    GeneticRisk: int = Form(...),
    PhysicalActivity: float = Form(...),
    AlcoholIntake: float = Form(...),
    CancerHistory: int = Form(...)
):
    user_input = {
        "Age": Age,
        "Gender": Gender,
        "BMI": BMI,
        "Smoking": Smoking,
        "GeneticRisk": GeneticRisk,
        "PhysicalActivity": PhysicalActivity,
        "AlcoholIntake": AlcoholIntake,
        "CancerHistory": CancerHistory
    }

    prediction, raw_votes = predict_majority(user_input)
    votes = [int(v) for v in raw_votes]


    return templates.TemplateResponse("form.html", {
        "request": request,
        "columns": columns,
        "prediction": prediction,
        "votes": votes,
        "slider_ranges": slider_ranges
    })
