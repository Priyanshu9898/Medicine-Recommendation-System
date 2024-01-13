from src.mlclassifier import logger
from src.mlclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.mlclassifier.config.configuration import ConfigurationManager
from src.mlclassifier.components.data_loader import DataLoader


# STAGE_NAME = "Data Ingestation"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(
#         f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

# except Exception as e:
#     logger.exception(e)
#     raise e



STAGE_NAME = "Data Loading and Processing"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    data_loader_config = config.get_data_loader_config()
    data_loader = DataLoader(config=data_loader_config)
    
    data, sym_des, precautions, workout, description, medications, diets = data_loader.loadDataset()
    
    X, Y, diseases_list, symptoms_dict = data_loader.processing(data)
    
    X_train, X_test, y_train, y_test = data_loader.splitDataset(X, Y)
    
    
    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
