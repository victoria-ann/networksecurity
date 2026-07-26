import sys,os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(self)->Pipeline:
        """This function initializes the KNN imputer object with the parameters defined in the training_pipline.py file and returns a Pipeline
        object with the KNNImputer object as the first step
        
        Args:
            cls: DataTransformation
        Returns:
            A Pipeline Object
        """
        logging.info("Entered the get_data_transformer_object method of DataTransformation class")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initiallized the KNNImputer object with the parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline = Pipeline([("Imputer",imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered the initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## Training dataframe
            # Remove the target feature from the independent features and replace -1 with 0 in the target feature
            imput_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            ## Testing dataframe
            imput_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            # Fit Transform Independent features for KNN imputer to replace the missing values and transform the independent features
            # Create a preprocessor object using the get_data_transformer_object method and fit it to the training independent features
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(imput_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(imput_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(imput_feature_test_df)

            # Create np arrays for train and test data by combining the transformed independent features and target features
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            save_object("final_model/preprocessor.pkl",preprocessor_object)
            
            # Preparing artifacts

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e