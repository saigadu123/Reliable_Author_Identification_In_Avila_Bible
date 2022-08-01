from Avila.pipeline.pipeline import Pipeline
from Avila.logger import logging
from Avila.exception import AvilaException
from Avila.config.configuration import Configuration
import os,sys
def main():
    try:
        logging.info(f"Main Pipeline started")
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        pipeline.start()
        logging.info(f"main function execution completed.") 
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__ == "__main__":
    main()