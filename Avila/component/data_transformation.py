from cgi import test
from msilib import schema
from sklearn import preprocessing
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.entity.config_entity import DataIngestionConfig, DataTransformationconfig
from Avila.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact,DataTransformationArtifact
import os,sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from Avila.constant import *
from Avila.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data

COLUMN_LOWER_MARGIN = "lower_margin"
COLUMN_UPPER_MARGIN = "upper_margin"

class FeatureGenerator(BaseEstimator,TransformerMixin):

    def __init__(self,add_total_margin_space=True,
                upper_margin_ix = 1,
                lower_margin_ix = 2,
                columns=None
                ):
        try:
            self.columns = columns 
            if self.columns is not None:
                self.lower_margin_ix = self.columns.index(COLUMN_LOWER_MARGIN)
                self.upper_margin_ix = self.columns.index(COLUMN_UPPER_MARGIN)
            self.lower_margin_ix = lower_margin_ix
            self.upper_margin_ix = upper_margin_ix
            self.add_total_margin_space = add_total_margin_space
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def fit(self,X,y=None):
        return self 

    def transform(self,X,y=None):
        try:
            if self.add_total_margin_space:
                total_margin_space = X.iloc[:,self.lower_margin_ix] + X.iloc[:,self.upper_margin_ix]
                generated_feature = np.c_[X,total_margin_space]
            else:
                generated_feature = np.c_[X]
            return generated_feature
        except Exception as e:
            raise AvilaException(e,sys) from e 


class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationconfig,
                data_ingestion_artifact:DataIngestionArtifact,
                data_validation_artifact:DataValidationArtifact):
        try:
            logging.info(f"{'='*20} Data Transformation Log started.{'='*60} \n \n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def get_data_transformer_object(self):
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema =  read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[DATASET_NUMERICAL_COLUMNS]

            num_pipeline = Pipeline(steps=[ 
                ('feature_generator',FeatureGenerator(add_total_margin_space=self.data_transformation_config.add_total_margin_space,columns=numerical_columns)),
                ('scaler',StandardScaler())
            ])

            logging.info(f"Numerical_columns: {numerical_columns}")

            preprocessing = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns) 
            ])
            return preprocessing 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def initiate_data_transformation(self):
        try:
            logging.info(f"Obtaining the Preprocessing object")
            preprocessed_obj = self.get_data_transformer_object()

            logging.info(f"getting training and testing file paths")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path 

            schema_file_path = self.data_validation_artifact.schema_file_path 
            logging.info(f"Loading training and testing datasets into pandas dataframes")

            train_df = load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_df = load_data(file_path = test_file_path,schema_file_path=schema_file_path)

            logging.info(f"Removing columns having multicolinearity")
            train_df.drop(columns=["modular_ratio/inter_linear_spacing"],axis=1)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]
            logging.info(f"splitting input feature and target feature from training and testing data")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing on training and testing input features")
            input_feature_train_arr = preprocessed_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessed_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]

            #file paths to save the transformed arrays
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace('.csv','.npz')
            test_file_name = os.path.basename(test_file_path).replace('.csv','.npz')

            transformed_train_file_path = os.path.join(transformed_train_dir,train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir,test_file_name)

            logging.info(f"Saving the transformed training and testing array")
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            logging.info(f"Saving the preprocessing object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path=preprocessing_object_file_path,obj=preprocessed_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed = True,
                message = "Data Transformation completed successfully",
                transformed_train_file_path = transformed_train_file_path,
                transformed_test_file_path = transformed_test_file_path,
                preprocessed_object_file_path = preprocessing_object_file_path
            )
            logging.info(f"Data Transformation Artifact :{data_transformation_artifact}")
            return data_transformation_artifact 
        except Exception as e:
            raise AvilaException(e,sys) from e 

    def __del__(self):
        logging.info(f"{'='*20} Data Transformation log Completed.{'='*60} \n \n")
