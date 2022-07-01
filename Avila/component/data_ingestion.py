from Avila.entity.config_entity import DataIngestionConfig
from Avila.entity.artifact_entity import DataIngestionArtifact
import os,sys
import tarfile
from Avila.logger import logging
from Avila.exception import AvilaException
from six.moves import urllib
import pandas as pd
import numpy as np