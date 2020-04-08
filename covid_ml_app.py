from flask import Flask, request, jsonify
from LGBM_regressor import *
import os
import json

app = Flask(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@app.route('/predict', methods=['POST'])
def predict():
    response = dict()
    try:
        days = request.form['num_days']
        response = model_predict(int(days))
        return jsonify(response)
    except Exception as e:
        response["success"] = False
        response["error"] = str(e)
        return jsonify(response)


@app.route('/fetch', methods=['GET'])
def fetch():
    response = dict()
    try:
        with open('data/prediction_result.json', 'r') as f:
            js = json.load(f)
        return js
    except Exception as e:
        response["success"] = False
        response["error"] = str(e)
        return jsonify(response)


app.run(host='0.0.0.0', port=6343, debug=True, use_reloader=False)