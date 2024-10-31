from flask import Flask, request, render_template
import pickle

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

# Route for homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route for Single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Retrieve input values from the form
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        # Preprocess the data using the loaded scaler
        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Make the prediction using the loaded model
        predict = model.predict(new_data)

        # Determine the result based on the prediction
        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        # Render the prediction.html template with the result
        return render_template('prediction.html', result=result)

    # In case of GET request or other methods, just render the homepage
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
