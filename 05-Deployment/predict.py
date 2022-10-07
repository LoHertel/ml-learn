import pickle

from flask import Flask, jsonify, request


with open('dv.bin', 'rb') as file:
    dv = pickle.load(file)

with open('model2.bin', 'rb') as file:
    model = pickle.load(file)

app = Flask('credit_card')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    result = {"prediction": float(y_pred),}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)