from cmath import log
import importlib
from operator import index
from pyexpat import model
import numpy as np
import yaml
from Avila.exception import AvilaException
import os,sys 

from collections import namedtuple
from typing import List
from Avila.logger import logging 
from sklearn.metrics import roc_auc_score

GRID_SEARCH_KEY = "grid_search"
MODULE_KEY = "module"
CLASS_KEY = "class"
PARAM_KEY = "params"
MODEL_SELECTION_KEY = "model_selection"
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",["model_serial_number","model","param_grid_search","model_name"])

GridSearchBestModel = namedtuple("GridSearchBestModel",["model_serial_number","model","best_model","best_parameters","best_score"])

BestModel = namedtuple("BestModel",["model_serial_number","model","best_model","best_parameters","best_score"])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",["model_name","model_object","train_accuracy","test_accuracy","model_accuracy","index_number"])




class ModelFactory:

    def __init__(self,model_config_path:str=None,):
        try:
            self.config:dict = ModelFactory.read_params(config_path=model_config_path)
            self.grid_search_cv_module = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_cv_name = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            self.models_initialization_config = dict(self.config[MODEL_SELECTION_KEY])
            self.initialized_model_list=None
            self.grid_searched_best_model_list=None 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    @staticmethod
    def update_property_of_class(instance_ref:object,property_data:dict):
        try:
            if not isinstance(property_data,dict):
                raise Exception("property data parameter required to be dictionary")
            print(property_data)
            for key,value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref,key,value)
            return instance_ref 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    @staticmethod
    def read_params(config_path:str)->dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    @staticmethod
    def class_for_name(module_name:str,class_name:str):
        try:
            module = importlib.import_module(module_name)
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module,class_name)
            return class_ref 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def execute_grid_search_operation(self,
                                    initialized_model:InitializedModelDetail,
                                    input_feature,
                                    output_feature)->GridSearchBestModel:
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,class_name=self.grid_search_cv_name)
            grid_search_cv = grid_search_cv_ref(estimator = initialized_model.model,
                                                param_grid = initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(instance_ref=grid_search_cv,property_data=self.grid_search_property_data)
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            grid_search_cv.fit(input_feature,output_feature)
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            grid_searched_best_model = GridSearchBestModel(model_serial_number = initialized_model.model_serial_number,
                                                        model = initialized_model.model,
                                                        best_model=grid_search_cv.best_estimator_,
                                                        best_parameters = grid_search_cv.best_params_,
                                                        best_score = grid_search_cv.best_score_)
            return grid_searched_best_model


        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_initialized_model_list(self)->List[InitializedModelDetail]:
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_object_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],class_name=model_initialization_config[CLASS_KEY])
                model = model_object_ref()
                
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,property_data=model_obj_property_data)
                
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}" 
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                    model = model,
                                                                    param_grid_search = param_grid_search,
                                                                    model_name=model_name)
                initialized_model_list.append(model_initialization_config)
                return initialized_model_list 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def initiate_best_parameter_search_for_initialized_model(self,
                                                            initialized_model:InitializedModelDetail,
                                                            input_feature,
                                                            output_feature):
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                        input_feature=input_feature,
                                                        output_feature=output_feature)
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def initiate_best_paramter_search_for_initialized_models(self,
                                                            initialized_models_list:List[InitializedModelDetail],
                                                            input_feature,
                                                            output_feature)->List[GridSearchBestModel]:
        try:
            grid_searched_best_models_list = []
            for initialized_model in initialized_models_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model = initialized_model,
                    input_feature = input_feature,
                    output_feature = output_feature 
                )
                grid_searched_best_models_list.append(grid_searched_best_model)
            return grid_searched_best_models_list 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_best_model(self,x,y,base_accuracy=0.6)->BestModel:
        try:
            logging.info(f"started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")

        except Exception as e:
            raise AvilaException(e,sys) from e 