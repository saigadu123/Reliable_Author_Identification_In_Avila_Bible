import os,sys
from Avila.exception import AvilaException
from Avila.util.util import load_object 
import pandas as pd
from Avila.logger import logging

class AuthorData:

    def __init__(self,Intercolumnar_distance:float,
                upper_margin:float,
                lower_margin:float,
                exploitation:float,
                row_number:float,
                modular_ratio:float,
                inter_linear_spacing:float,
                weight:float,
                peak_number:float,
                modular_per_inter:float,
                Class: object =None):
        try:
            self.Intercolumnar_distance=Intercolumnar_distance,
            self.upper_margin = upper_margin,
            self.lower_margin = lower_margin,
            self.exploitation = exploitation,
            self.row_number = row_number,
            self.modular_ratio = modular_ratio,
            self.inter_linear_spacing = inter_linear_spacing,
            self.weight = weight,
            self.peak_number = peak_number
            self.modular_per_inter = modular_per_inter
            self.Class = Class  
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_avila_input_data_frame(self)->pd.DataFrame:
        try:
            Avila_input_dict = self.get_Avila_data_as_dict()
            return pd.DataFrame(Avila_input_dict,index=[0])
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_Avila_data_as_dict(self):
        try:
            input_data = {
                "Intercolumnar_distance":self.Intercolumnar_distance,
                "upper_margin":self.upper_margin,
                "lower_margin":self.lower_margin,
                "exploitation":self.exploitation,
                "row_number":self.row_number,
                "modular_ratio":self.modular_ratio,
                "inter_linear_spacing":self.inter_linear_spacing,
                "weight":self.weight,
                "peak_number":self.peak_number,
                "modular_ratio/inter_linear_spacing": self.modular_per_inter
            }
            return input_data 
        except Exception as e:
            raise AvilaException(e,sys) from e 

class AuthorPredictor:

    def __init__(self,model_dir:str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int,os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir,f"{max(folder_name)}")
            filename = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir,filename)
            return latest_model_path
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def predict(self,X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            conversion_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'W',10:'X',11:'Y'}
            logging.info(f"The columns to predict are {X} and type of x is{type(X)}")
            Author_name = model.predict(X)
            return conversion_dict[Author_name[0]]
        except Exception as e:
            raise AvilaException(e,sys) from e 

    
                