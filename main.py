import webbrowser
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, Form
import threading
import time
from model import predict_majority

app = FastAPI()
templates = Jinja2Templates(directory="templates")

columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

def open_browser():
    time.sleep(1)  # wait for server to start
    webbrowser.open("http://127.0.0.1:8000")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=open_browser).start()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "columns": columns, "prediction": None})

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

    prediction, votes = predict_majority(user_input)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "columns": columns,
        "prediction": prediction,
        "votes": votes
    })
