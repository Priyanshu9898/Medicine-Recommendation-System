from src.mlclassifier import logger
from src.mlclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from src.kidneyDiseaseClassifier.pipeline.stage_02_model_building import ModelBuildingPipeline
# from src.kidneyDiseaseClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
# from kidneyDiseaseClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestation"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
