from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_tranformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys

if __name__=="__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiated the data ingestion component")
        dataingestionartifact = data_ingestion.intitiate_data_ingestion()
        logging.info("Data ingestion component completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiated the data validation component")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation component completed")
        print(data_validation_artifact)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiated the data transformation component")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation component completed")
        print(data_transformation_artifact)
        logging.info("Model training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config,data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys) from e