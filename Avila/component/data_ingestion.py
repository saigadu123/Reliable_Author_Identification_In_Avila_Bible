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
            logging.info(f"{'='*20} data ingestion log started.{'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def dowload_Avila_data(self)->str:
        try:
            # Take the url formed in configuration module
            download_url = self.data_ingestion_config.dataset_download_url

            #url location to download tgz file
            tgz_download_dir  = self.data_ingestion_config.tgz_download_dir
            if os.path.exists(tgz_download_dir):
                os.remove(tgz_download_dir)
            os.makedirs(tgz_download_dir,exist_ok=True)
            Avila_file_name = os.path.basename(download_url)
            tgz_file_path = os.path.join(tgz_download_dir,Avila_file_name)
            logging.info(f"Download file from [{download_url}] into: [{tgz_file_path}]")
            urllib.request.urlretrieve(download_url,tgz_file_path)
            logging.info(f"File is successfully downloaded into [{Avila_file_name}]")
            print(f"Hii tgz file path is {tgz_file_path} ")
            return tgz_file_path
        except Exception as e:
            raise AvilaException(e,sys) from e

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir [{raw_data_dir}]")
            with tarfile.open(tgz_file_path) as Avila_tgz_file_obj:
                Avila_tgz_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction Completed")
        except Exception as e:
            raise AvilaException(e,sys) from e


    def split_as_train_test(self)->DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            filename = os.listdir(raw_data_dir)[0]
            Avila_file_path = os.path.join(raw_data_dir,filename)
            logging.info(f"Reading CSV file from [{Avila_file_path}]")
            Avila_data_frame = pd.read_csv(Avila_file_path)
            Avila_data_frame.columns = ["Intercolumnar_distance","upper_margin","lower_margin","exploitation","row_number","modular_ratio","inter_linear_spacing","weight","peak_number","modular_ratio","class"]
            logging.info("Splitting Dataset into train and test")
            strat_train_set = None
            strat_test_set = None
    
            split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
            for train_index,test_index in split.split(Avila_data_frame,Avila_data_frame['class']):
                strat_train_set = Avila_data_frame.loc[train_index].drop(["class"],axis=1)
                strat_test_set = Avila_data_frame.loc[test_index].drop(["class"],axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,filename)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,filename)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting train file into [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingested_test_dir,test_file_path,exist_ok=True)
                logging.info(f"Exporting test file into [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            
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
            tgz_file_path = self.dowload_Avila_data()
            self.extract_tgz_file(tgz_file_path)
            return self.split_as_train_test()
        except Exception as e:
            raise AvilaException(e,sys) from e