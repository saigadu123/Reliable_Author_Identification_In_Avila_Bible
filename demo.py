from Avila.pipeline.pipeline import Pipeline
from Avila.logger import logging
from Avila.exception import AvilaException

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__ == "__main__":
    main()