from setuptools import setup,find_packages
from typing import List

#Declaring variables for a setup function

PROJECT_NAME = "Avila_Bible_predictor"
VERSION = "0.0.3"
AUTHOR = "sai krishna"
DESCRIPTION = "This is the Avila-Bible prediction project"

REQUIREMENT_FILE_NAME = "requirements.txt"

def get_requirements_list()->List[str]:
    """
     This function is going to return the list of requirement mention in requirements.txt file
     
     It return the libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_file.readlines().remove('-e .')

    
setup(
    name = PROJECT_NAME,
    version = VERSION,
    author = AUTHOR,
    description = DESCRIPTION,
    packages = find_packages(),
    install_requires = get_requirements_list()
)