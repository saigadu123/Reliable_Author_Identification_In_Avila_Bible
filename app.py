from flask import Flask
from Avila.logger import logging

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return "Avila-Bible project"


if __name__ == "__main__":
    app.run(debug=True)