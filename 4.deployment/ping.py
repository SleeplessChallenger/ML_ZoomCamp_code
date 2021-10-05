from flask import Flask, request, jsonify
import numpy as np
from pickle import dump, load


app = Flask('model_churn')


with open('model_C=1.0.bin', 'rb') as file:
	dv, model = load(file)


@app.route('/predict', methods=['POST'])
def ping():
	# take body and convert into dict()
	data = request.get_json()
	y_pred = predict(data)
	churn_decision = y_pred >= 0.5

	# convert results from numpy types
	result = {
		'pred.': float(y_pred),
		'churn': bool(churn_decision)
	}

	return jsonify(result)

def predict(customer):
	X = dv.transform([customer])
	y_pred = model.predict_proba(X)[0, 1]
	return y_pred


if __name__ == '__main__':
	'''
	when using gunicron this chunk
	won't be executed
	'''
	app.run(debug=True, host='0.0.0.0', port=7000)

# gunicorn --bind 0.0.0.0:7000 ping:app