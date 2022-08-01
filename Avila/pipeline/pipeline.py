from tkinter import E 
import os,sys
from Avila.component.model_pusher import ModelPusher
from Avila.config.configuration import Configuration
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationconfig
from Avila.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelEvaluationArtifact,ModelTrainerArtifact
from Avila.component.data_ingestion import DataIngestion
from Avila.component.data_validation import DataValidation
from Avila.component.data_transformation import DataTransformation 
from Avila.component.model_trainer import ModelTrainer
from Avila.component.model_evaluation import ModelEvaluation
from datetime import datetime
from threading import Thread
import pandas as pd
from Avila.constant import EXPERIMENT_DIR_NAME,EXPERIMENT_FILE_NAME
from collections import namedtuple
import uuid
import numpy as np

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                        "running_status", "start_time", "stop_time", "execution_time", "message",
                                        "experiment_file_path", "accuracy", "is_model_accepted"])

class Pipeline(Thread):
    experiment:Experiment = Experiment(*([None]*11))
    experiment_file_path = None 

    def __init__(self,config:Configuration)->None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir,exist_ok=True)
            Pipeline.experiment_file_path = os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME,EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False,name="pipeline")
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

    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation = DataTransformation(data_transformation_config=self.config.get_data_transformation_config(),
                                                    data_ingestion_artifact = data_ingestion_artifact,
                                                    data_validation_artifact = data_validation_artifact)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer = ModelTrainer(model_trainer_config = self.config.get_model_trainer_config(),data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer() 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_model_evaluation(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact,model_trainer_artifact:ModelTrainerArtifact):
        try:
            model_evaluation = ModelEvaluation(model_evaluation_config=self.config.get_model_evaluation_config(),
                                                data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_artifact=data_validation_artifact,
                                                model_trainer_artifact=model_trainer_artifact) 
            return model_evaluation.initiate_model_evaluation()
            
            
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(model_pusher_config = self.config.get_model_pusher_config(),model_evaluation_artifact=model_evaluation_artifact)
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info(f"Pipeline is already running")
                return Pipeline.experiment 
            logging.info(f"Pipeline is starting")
            experiment_id = str(uuid.uuid4())
            Pipeline.experiment = Experiment(experiment_id=experiment_id,
                                            initialization_timestamp=self.config.time_stamp,
                                            artifact_time_stamp=self.config.time_stamp,
                                            running_status=True,
                                            start_time = datetime.now(),
                                            stop_time=np.nan,
                                            execution_time = np.nan,
                                            experiment_file_path=Pipeline.experiment_file_path,
                                            is_model_accepted=np.nan,
                                            message = "Pipeline has started",
                                            accuracy = np.nan
                                            )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact = data_ingestion_artifact,data_validation_artifact = data_validation_artifact,model_trainer_artifact = model_trainer_artifact)

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info(f"Trained Model Rejected")
            
            logging.info(f"Pipeline Completed")

            stop_time = datetime.now()

            Pipeline.experiment = Experiment(experiment_id = experiment_id,
                                            initialization_timestamp=self.config.time_stamp,
                                            artifact_time_stamp=self.config.time_stamp,
                                            running_status=False,
                                            start_time = Pipeline.experiment.start_time,
                                            stop_time=stop_time,
                                            execution_time = stop_time-Pipeline.experiment.start_time,
                                            experiment_file_path=Pipeline.experiment_file_path,
                                            message = "Pipeline has been Completed",
                                            is_model_accepted=model_evaluation_artifact.is_model_accepted,
                                            accuracy = model_trainer_artifact.model_accuracy
                                            )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
        except Exception as e:
            raise AvilaException(e,sys) from e 
    
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict:dict = {key:[value] for key,value in experiment_dict.items()}
                experiment_dict.update({
                    "created_time_stamp":[datetime.now()],
                    "experiment_file_path":[os.path.basename(Pipeline.experiment.experiment_file_path)]
                })
                experiment_report = pd.DataFrame(experiment_dict)
                os.makedirs(os.path.dirname(Pipeline.experiment_file_path),exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path,index=False,header=False,mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path,index=False,header=False,mode="w")
            else:
                print("First start experiment")
        except Exception as e:
            raise AvilaException(e,sys) from e 

    @classmethod
    def get_experiments_status(cls,limit:int = 5)->pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1*int(limit)
                return df[limit:]
            else:
                return pd.DataFrame() 
        except Exception as e:
            raise AvilaException(e,sys) from e 



