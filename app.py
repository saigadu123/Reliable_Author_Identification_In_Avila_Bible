from flask import Flask
from Avila.logger import logging
from Avila.exception import AvilaException
import sys

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
       raise Exception("We are raising custom Exception")
    except Exception as e:
        avila = AvilaException(e,sys)
        logging.info(avila.error_message)

    logging.info("My own logger project")
    return "Avila-Bible project ineuron internship"


if __name__ == "__main__":
    app.run(debug=True)