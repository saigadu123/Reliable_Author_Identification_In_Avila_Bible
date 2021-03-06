from tkinter import E 
import os,sys
from Avila.config.configuration import Configuration
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.entity.config_entity import DataIngestionConfig,DataValidationConfig
from Avila.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from Avila.component.data_ingestion import DataIngestion
from Avila.component.data_validation import DataValidation


class Pipeline:

    def __init__(self,config:Configuration=Configuration())->None:
        try:
            self.config = config
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config = self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise AvilaException(e,sys)

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config = self.config.get_data_validation_config(),data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_data_transformation(self):
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_model_trainer(self):
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_model_evaluation(self):
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_model_pusher(self):
        try:
            pass
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise AvilaException(e,sys) from e 




