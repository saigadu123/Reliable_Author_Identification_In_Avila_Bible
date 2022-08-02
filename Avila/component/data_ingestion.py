from Avila.entity.config_entity import DataIngestionConfig
from Avila.entity.artifact_entity import DataIngestionArtifact
import os,sys
import tarfile
from Avila.logger import logging
from Avila.exception import AvilaException
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20} data ingestion log started.{'='*20} \n \n")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def dowload_Avila_data(self)->str:
        try:
            # Take the url formed in configuration module
            download_url = self.data_ingestion_config.dataset_download_url
            Avila_file_name = os.path.basename(download_url)
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            data_frame = pd.read_csv(download_url)
            logging.info(f"Data Frame is created")
            os.makedirs(raw_data_dir,exist_ok=True)
            csv_file_path = os.path.join(raw_data_dir,Avila_file_name)
            data_frame.to_csv(csv_file_path,index=False)
            logging.info(f"CSV file is downloaded into Raw data Directory")
        except Exception as e:
            raise AvilaException(e,sys) from e

   


    def split_as_train_test(self)->DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            filename = os.listdir(raw_data_dir)[0]
            Avila_file_path = os.path.join(raw_data_dir,filename)
            logging.info(f"Reading CSV file from [{Avila_file_path}]")
            Avila_data_frame = pd.read_csv(Avila_file_path)

            
            logging.info(f"Removing columns having multicolinearity")
            Avila_data_frame.drop(columns=["modular_ratio/inter_linear_spacing"])
            logging.info(f"columns in Avila dataframe are {Avila_data_frame.columns}")
            logging.info("Splitting Dataset into train and test")
            strat_train_set = None
            strat_test_set = None
            columns = ["Intercolumnar_distance","upper_margin","lower_margin","exploitation","row_number","modular_ratio","inter_linear_spacing","weight","peak_number","modular_ratio/inter_linear_spacing","Class"]
            split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
            for train_index,test_index in split.split(Avila_data_frame,Avila_data_frame['Class']):
                strat_train_set = Avila_data_frame.loc[train_index]
                strat_test_set = Avila_data_frame.loc[test_index]

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,filename)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,filename)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting train file into [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False,columns=columns)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
                logging.info(f"Exporting test file into [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False,columns=columns)
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = train_file_path,
                test_file_path = test_file_path,
                is_ingested = True,
                message = f"Data Ingestion completed Successfiully"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise AvilaException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
          
            self.dowload_Avila_data()
            return self.split_as_train_test()
        except Exception as e:
            raise AvilaException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*30} Data Ingestion log completed.{'='*60}\n \n ")