from this import d
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.entity.config_entity import ModelPusherConfig
from Avila.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
import os,sys
import shutil

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
                    model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def export_model(self)->ModelPusherArtifact:
        try:
            model_evaluated_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path
            model_file_name = os.path.basename(model_evaluated_file_path)
            export_file_path = os.path.join(export_dir,model_file_name)
            os.makedirs(export_dir,exist_ok=True)
            shutil.copy(src=model_evaluated_file_path,dst=export_file_path)

            logging.info(
                f"Trained model: {model_evaluated_file_path} is copied in export dir:[{export_file_path}]")
            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,export_model_file_path=export_file_path)
            return model_pusher_artifact
        except Exception as e:
            raise AvilaException(e,sys) from e 
    
    def initiate_model_pusher(self):
        try:
            self.export_model()
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")