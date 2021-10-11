from flask import Flask, request, jsonify
import numpy as np
from pickle import load, dump


app = Flask('prediction_service')


@app.route('/predict_results', methods=['POST'])
def receive_prediction():
	data = request.get_json()
	pred, churn = prediction(data)

	result = {
		'pred.': float(pred),
		'decision': bool(churn)
	}

	return jsonify(result)

def prediction(data):
	dv, model = unpickle_data()
	X = dv.transform([data])
	y_pred = model.predict_proba(X)[0, 1]
	return y_pred, y_pred >= 0.5


def unpickle_data():
	with open('model_trained.bin', 'rb') as file:
		dv, model = load(file)

	return dv, model


if __name__ == '__main__':
	app.run(debug=True, host='localhost', port=7070)
