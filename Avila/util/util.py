import sys
import os
import yaml
from Avila.exception import AvilaException

def read_yaml_file(file_path:str)->dict:
    
    try:
        with open(file_path,'rb') as yaml_file:
           return  yaml.safe_load(yaml_file)
    except Exception as e:
        raise AvilaException(e,sys) from e

