import os,sys
from tkinter import E
from Avila.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationconfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig, TrainingPipelineConfig
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.constant import *
from Avila.util.util import read_yaml_file

class Configuration:

    def __init__(self,config_file_path:str = CONFIG_FILE_PATH, curr_time_stamp:str = CURRENT_TIME_STAMP)->None:
        self.config = read_yaml_file(file_path=config_file_path)
        self.training_pipeline_config = self.get_training_pipeline_config()
        self.time_stamp = curr_time_stamp

    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir = os.path.join(artifact_dir,
            DATA_INGESTION_ARTIFACT_DIR,
            self.time_stamp
            )
            data_ingestion_info = self.config[DATA_INGESTION_CONFIG_KEY]
            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            tgz_download_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY]
            )
            raw_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
            )
            ingested_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            )
            ingested_train_dir = os.path.join(
                ingested_dir,
                data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]
            )
            ingested_test_dir = os.path.join(
                ingested_dir,
                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]
            )
            data_ingestion_config = DataIngestionConfig(
                dataset_download_url = dataset_download_url,
                tgz_download_dir = tgz_download_dir,
                raw_data_dir = raw_data_dir,
                ingested_train_dir = ingested_train_dir,
                ingested_test_dir = ingested_test_dir
            )
            return data_ingestion_config
        except Exception as e:
            raise AvilaException(e,sys) from e

    def get_data_validation_config(self)->DataValidationConfig:
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_data_transformation_config(self)->DataTransformationconfig:
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training_pipeline_config:{training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise AvilaException(e,sys) from e 