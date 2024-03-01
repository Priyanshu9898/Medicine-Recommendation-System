from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
from src.mlclassifier.config.configuration import ConfigurationManager
from src.mlclassifier.components.data_loader import DataLoader
from src.mlclassifier.components.prediction import Prediction
from src.mlclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlclassifier.components.model_training import ModelTraining
from src.mlclassifier.components.model_evaluation import ModelEvaluation
from src.mlclassifier.pipeline.prediction_pipeline import PredictionPipeline
from src.mlclassifier import logger
import numpy as np

app = FastAPI()

# CORS configuration
allowed_origins = ["https://meta-doctor.vercel.app", "https://metadoctor.vercel.app/", "https://meta-doctor.vercel.app", "https://meta-doctor.vercel.app/", "https://metadoctor-git-main-priyanshumalaviya9210-gmailcom.vercel.app/", "https://metadoctorhelper.vercel.app", "http://localhost:3000",
    "http://127.0.0.1:3000", "https://metadoctorhelper.vercel.app", "https://metadoctor-priyanshumalaviya9210-gmailcom.vercel.app/", "https://meta-doctor.vercel.app", "https://meta-doctor-nwxeblfpx-priyanshumalaviya9210-gmailcom.vercel.app", "https://meta-doctor-git-main-priyanshumalaviya9210-gmailcom.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load datasets
logger.info("Starting to load datasets.")
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")
data = pd.read_csv("Training.csv")
logger.info("All datasets loaded.")

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Medicine Recommendation System Backend</h1> <br /> Go to Swagger Docs: <a href='/docs'>Link</a>"

def clean_data(data):
    """
    Recursively replace NaN and infinite values in the response data,
    so it can be JSON serialized without errors.
    """
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif isinstance(data, (float, np.float32, np.float64)) and (np.isnan(data) or np.isinf(data)):
        return None  # Replace NaN or inf with None
    return data

@app.post("/predict")
async def predict(request: Request):
    try:
        input_data = await request.json()
        
        # print(input_data)
        
        input_symptoms = input_data.get('symptoms', [])
        
        prediction_pipeline = PredictionPipeline(symptoms=input_symptoms, description=description, precautions=precautions,
                                                 medications=medications, diets=diets, workout=workout, data=data, sym_des=sym_des)
        results = prediction_pipeline.main()
        
        print("*"*100, results)
        
        clean_results = clean_data(results)
        
        return JSONResponse(content=clean_results)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=5000)


# from flask import Flask, request, jsonify
# from flask_cors import cross_origin
# from flask_swagger_ui import get_swaggerui_blueprint
# from flask_cors import CORS
# from src.mlclassifier.config.configuration import ConfigurationManager
# from src.mlclassifier.components.data_loader import DataLoader
# from src.mlclassifier.components.prediction import Prediction
# from src.mlclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from src.mlclassifier.components.model_training import ModelTraining
# from src.mlclassifier.components.model_evaluation import ModelEvaluation
# from src.mlclassifier import logger
# from src.mlclassifier.pipeline.prediction_pipeline import PredictionPipeline
# import pandas as pd
# app = Flask(__name__)


# # Define allowed origins for CORS
# allowed_origins = ["https://metadoctor.vercel.app/", "https://meta-doctor.vercel.app", "https://meta-doctor.vercel.app/", "https://metadoctor-git-main-priyanshumalaviya9210-gmailcom.vercel.app/",  "https://metadoctorhelper.vercel.app", "http://localhost:3000",
#                    "http://127.0.0.1:3000", "https://metadoctorhelper.vercel.app", "https://metadoctor-priyanshumalaviya9210-gmailcom.vercel.app/", "https://meta-doctor.vercel.app", "https://meta-doctor-nwxeblfpx-priyanshumalaviya9210-gmailcom.vercel.app", "https://meta-doctor-git-main-priyanshumalaviya9210-gmailcom.vercel.app"]

# CORS(app, origins=allowed_origins)

# SWAGGER_URL = '/swagger'
# API_URL = '/static/swagger.json'
# swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
# app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# logger.info("Starting to load datasets.")


# sym_des = pd.read_csv("symtoms_df.csv")
# logger.info("Symptoms dataset loaded.")

# precautions = pd.read_csv("precautions_df.csv")
# logger.info("Precautions dataset loaded.")

# workout = pd.read_csv("workout_df.csv")
# logger.info("Workout dataset loaded.")

# description = pd.read_csv("description.csv")
# logger.info("Description dataset loaded.")

# medications = pd.read_csv('medications.csv')
# logger.info("Medications dataset loaded.")

# diets = pd.read_csv("diets.csv")
# logger.info("Diets dataset loaded.")

# data = pd.read_csv("Training.csv")
# logger.info("Training dataset loaded.")


# @app.after_request
# def after_request(response):
#     header = response.headers
#     log_message = f"Access-Control-Allow-Origin: {header.get('Access-Control-Allow-Origin')}"
#     logger.info(log_message)
#     return response

# @app.route('/predict', methods=['OPTIONS'])
# @cross_origin(origins=allowed_origins)
# def predict_options():
#     return jsonify({'message': 'OPTIONS request allowed'}), 200


# @app.route('/predict', methods=['POST'])
# @cross_origin(origins=allowed_origins)
# def predict():
#     try:
#         input_data = request.json

#         # description, precautions, medications, diets, workout, data, sym_des

#         print(input_data)
#         input_symptoms = input_data.get('symptoms', [])

#         # Initialize the PredictionPipeline
#         prediction_pipeline = PredictionPipeline(symptoms=input_symptoms, description=description, precautions=precautions,
#                                                  medications=medications, diets=diets, workout=workout, data=data, sym_des=sym_des)
#         results = prediction_pipeline.main()

#         # print(type(results['Workout']))

#         # print((results))

#         return jsonify(results), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.route('/')
# @cross_origin(origins=allowed_origins)
# def index():
#     return "<h1>Medicine Recommendation System Backend</h1> <br /> Go to Swagger Docs: <a href='/swagger'>Link</a>"


# if __name__ == '__main__':
#     app.run(debug=False, host='0.0.0.0', port=5000)
