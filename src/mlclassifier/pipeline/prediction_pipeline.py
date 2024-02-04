from src.mlclassifier.config.configuration import ConfigurationManager
from src.mlclassifier.components.prediction import Prediction
from src.mlclassifier import logger
from src.mlclassifier.components.data_loader import DataLoader

STAGE_NAME = "Prediction"


class PredictionPipeline:
    def __init__(self, symptoms):
        self.symptoms = symptoms

    def main(self):
        config = ConfigurationManager()
        
        data_loader_config = config.get_data_loader_config()
        data_loader = DataLoader(config=data_loader_config)
    
        prediction_config = config.get_model_prediction_config()
        trained_model_filename  = prediction_config.best_model
        
        print(trained_model_filename)
        
        pipeline = Prediction(data_loader, trained_model_filename)
        pipeline.load_model()
        # self.symptoms = ["itching","skin_rash","nodal_skin_eruptions"]
        results = pipeline.predict(self.symptoms)
        
        # print(results)

        # Print the results
        # for key, value in results.items():
        #     logger.info(f"{key}: {value}")
        
        return results

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e