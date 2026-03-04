from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv('diabetes.csv')
X = df[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','Glucose','DiabetesPedigreeFunction','Age']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [
            float(request.form.get("Pregnancies")),
            float(request.form.get("Glucose")),
            float(request.form.get("BloodPressure")),
            float(request.form.get("SkinThickness")),
            float(request.form.get("Insulin")),
            float(request.form.get("BMI")),
            float(request.form.get("DiabetesPedigreeFunction")),
            float(request.form.get("Age")),
        ]

        prediction = model.predict([input_features])[0]
        if prediction <= 0.25 : 
            pred = 'low'
        elif prediction >= 0.25 and prediction <= 50 : 
            pred  = 'average'
        else : 
            pred = 'high'

        return render_template("index.html", prediction_text=pred)

    except:
        return render_template("index.html", prediction_text="Error: Please enter valid numeric values.")


if __name__ == "__main__":
    app.run(debug=True)