from flask import Flask, request, jsonify
from flask_cors import cross_origin
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from src.mlclassifier.config.configuration import ConfigurationManager
from src.mlclassifier.components.data_loader import DataLoader
from src.mlclassifier.components.prediction import Prediction
from src.mlclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlclassifier.components.model_training import ModelTraining
from src.mlclassifier.components.model_evaluation import ModelEvaluation
from src.mlclassifier import logger
from src.mlclassifier.pipeline.prediction_pipeline import PredictionPipeline

application=Flask(__name__)

app=application

# Define allowed origins for CORS
allowed_origins = ["http://localhost:3000",
                   "http://127.0.0.1:3000", "https://metadoctorhelper.vercel.app"]

CORS(app)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route('/predict', methods=['POST'])
@cross_origin(origins=allowed_origins)
def predict():
    try:
        input_data = request.json

        print(input_data)
        input_symptoms = input_data.get('symptoms', [])

        # Initialize the PredictionPipeline
        prediction_pipeline = PredictionPipeline(symptoms=input_symptoms)
        results = prediction_pipeline.main()

        # print(type(results['Workout']))

        # print((results))

        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
@cross_origin(origins=allowed_origins)
def index():
    return "<h1>Medicine Recommendation System Backend</h1> <br /> Go to Swagger Docs: <a href='/swagger'>Link</a>"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
