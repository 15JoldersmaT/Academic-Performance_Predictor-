import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
data = pd.read_csv('Student_Performance.csv')

data['Extracurricular Activities'] =label_encoder.fit_transform(data['Extracurricular Activities'])

#reshape(-1,1) ?
x = data[['Hours Studied', 'Extracurricular Activities', 'Sleep Hours']]
y = data['Performance Index']




print(x)


def train(x, y):
    model = LinearRegression().fit(x,y)
    return model

model = train(x.values,y)

def test():
    try:
        studyHours = float(input('Enter Hours studied: '))
        extraCur = input('Does the student take part in extra curricular activities? (Yes or No) : ')
        sleepHours = float(input('Enter Avg Hours student sleeps: '))

        extraCurT = label_encoder.transform([extraCur])[0]
        x_new = np.array([[studyHours, extraCurT, sleepHours]])
        y_new = model.predict(x_new)
        print ('Predicted Student Performance : '  + str(y_new))
        print('#######################NEW TEST###################')
        test()
    except:
        print ("Invalid input")
        test()





test()


