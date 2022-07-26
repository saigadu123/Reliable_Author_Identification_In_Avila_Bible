from cgi import test
from sklearn import preprocessing
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.entity.config_entity import DataTransformationconfig
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

