from tkinter import E 
from Avila.logger import logging
from Avila.exception import AvilaException
import os,sys
from Avila.entity.config_entity import DataValidationConfig
from Avila.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json 

class DataValidation:

    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_train_test_df(self):
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            return train_df,test_df
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def is_train_and_test_file_exists(self):
        try:

            logging.info(f"{'='*20} Data Validation Log started.{'='*60} \n \n")
            logging.info("Checking if training and testing file exists or not")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exists = os.path.exists(train_file_path)
            is_test_file_exists = os.path.exists(test_file_path)
            
            is_available = is_train_file_exists and is_test_file_exists
            
            logging.info(f"Is training and testing file exists?-> {is_available}")

            if not is_available:
                training_filename = self.data_ingestion_artifact.train_file_path
                test_filename = self.data_ingestion_artifact.test_file_path
                message = f"Training file: [{training_filename}] and test file [{test_filename}] are not present"
                raise Exception(message)
            return is_available
        
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False
            train_df,test_df = self.get_train_test_df()
            profile_dict = dict()
            logging.info(f"Checking no of rows and columns")
            no_of_rows = train_df.shape[0]
            no_of_columns = train_df.shape[1]
            profile_dict["Rows"] = no_of_rows
            profile_dict["Columns"] = no_of_columns

            column_names = list(train_df.columns)
            profile_dict["Column_names"] = column_names 
            profile_file_path = self.data_validation_config.profile_file_path
            profile_report_dir = os.path.dirname(profile_file_path)
            os.makedirs(profile_report_dir,exist_ok=True)
            with open(profile_file_path,"w") as profile_file:
                json.dump(profile_dict,profile_file,indent=6)
            validation_status = True 
            return profile_file_path 

        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df,test_df = self.get_train_test_df()
            profile.calculate(train_df,test_df)
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            #report_dir = os.path.dirname(report_file_path)
            with open(report_file_path,"w") as report_file:
                json.dump(report,report_file,indent=6)
            return report_file_path

        except Exception as e:
            raise AvilaException(e,sys) from e 

    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df,test_df = self.get_train_test_df()
            dashboard.calculate(train_df,test_df)
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)
            dashboard.save(report_page_file_path)
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def is_data_drift_found(self)->bool:
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            profile_report = self.validate_dataset_schema()
            return True 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            self.is_train_and_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                profile_path= self.data_validation_config.profile_file_path,
                report_file_path = self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated = True,
                message = "Data Validation Completed Successfully"
            )
            return data_validation_artifact 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def __del__(self):
        logging.info(f"{'='*20} Data Validation Log completed.{'='*60} \n \n")
    


