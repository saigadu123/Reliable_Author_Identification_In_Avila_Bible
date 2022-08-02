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


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "Avila"
SAVED_MODELS_DIR_NAME = "saved_models"
LOG_DIR = os.path.join(ROOT_DIR,LOG_FOLDER_NAME)
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
PIPELINE_DIR = os.path.join(ROOT_DIR,PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR,SAVED_MODELS_DIR_NAME)

from Avila.logger import get_log_dataframe

AVILA_DATA_KEY = "avila_data"
AUTHOR_VALUE_KEY = "author_name"

app = Flask(__name__)

@app.route('/artifact', defaults={'req_path': 'Avila'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("Avila", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)
    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
            "artifact" in os.path.join(abs_path, file_name)}
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)

@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(curr_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        AVILA_DATA_KEY: None,
        AUTHOR_VALUE_KEY: None
    }
    if request.method == 'POST':
        Intercolumnar_distance = float(request.form['Intercolumnar_distance'])
        upper_margin = float(request.form["upper_margin"])
        lower_margin = float(request.form["lower_margin"])
        exploitation = float(request.form["exploitation"])
        row_number = float(request.form["row_number"])
        modular_ratio = float(request.form["modular_ratio"])
        inter_linear_spacing = float(request.form["inter_linear_spacing"])
        weight = float(request.form["weight"])
        peak_number = float(request.form["peak_number"])
        modular_per_inter = float(request.form["modular_per_inter"])

        author_data = AuthorData(Intercolumnar_distance=Intercolumnar_distance,
                                upper_margin = upper_margin,
                                lower_margin = lower_margin,
                                exploitation = exploitation,
                                row_number = row_number,
                                modular_ratio = modular_ratio,
                                inter_linear_spacing = inter_linear_spacing,
                                weight = weight,
                                peak_number = peak_number,
                                modular_per_inter = modular_per_inter
                                )
        avila_df = author_data.get_avila_input_data_frame()
        avila_predictor = AuthorPredictor(model_dir=MODEL_DIR)
        author_name = avila_predictor.predict(X=avila_df)
        context = {
            AVILA_DATA_KEY: author_data.get_housing_data_as_dict(),
            AUTHOR_VALUE_KEY: author_name,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)

@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)
    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)

@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)
            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)
        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)

@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

         # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)