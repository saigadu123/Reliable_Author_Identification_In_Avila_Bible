from flask import Flask, request 
from Avila.logger import logging
from Avila.exception import AvilaException
import sys,os
import pip
from Avila.util.util import read_yaml_file,write_yaml_file
from matplotlib.style import context
import json
from Avila.config.configuration import Configuration
from Avila.constant import CONFIG_DIR, ROOT_DIR,get_current_time_stamp
from Avila.pipeline.pipeline import Pipeline 
from Avila.entity.Author_predictor import AuthorData,AuthorPredictor
from flask import send_file,abort,render_template




app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)



if __name__ == "__main__":
    app.run()