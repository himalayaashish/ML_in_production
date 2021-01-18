import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn import preprocessing
app = Flask(__name__)
model = pickle.load(open('model_flood_sc.pkl', 'rb'))
model = pickle.load(open('model_flood.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Number of deaths will be closely around {}'.format(int(output)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    #scale = preprocessing.Normalizer()
    #scale_df = scale.fit_transform(data)
    #prediction = model.predict([np.array(list(scale_df.values()))])
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)