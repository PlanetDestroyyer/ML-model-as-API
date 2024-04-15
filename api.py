from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

class model_input(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int



with open('classifier.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/data_pred')
def data_pred(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)
    preg = input_dict['Pregnancies']
    glu = input_dict['Glucose']
    bp = input_dict['BloodPressure']
    skinT = input_dict['SkinThickness']
    Insu = input_dict['Insulin']   
    bmi = input_dict['BMI'] 
    dp = input_dict['DiabetesPedigreeFunction']
    age = input_dict['Age']

    input_list = [preg,glu,bp,skinT,Insu,bmi,dp,age]

    prediction = model.predict([input_list])

    if prediction[0] == 0:
        return "The Person is Not Diabetic"
    else :
        return "The Person is Diabetic"
    




