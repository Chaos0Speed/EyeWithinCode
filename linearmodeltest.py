import numpy as np
import pandas as pd
import  tensorflow  as  tf
from tensorflow.keras import layers, models

def z_score_scaling(data):
    return (data - data.mean(axis=0)) / data.std(axis=0),data.mean(axis=0),data.std(axis=0)

if('model.keras') in tf.io.gfile.listdir('.'):
    model = models.load_model('model.keras')
else:
    exit()
print(model.summary())
test = pd.read_excel('Test data.xlsx')
test['internet'] = test['internet'].map({'no':1,'yes':0})
test['sex'] = test['sex'].map({'M':1,'F':-1})
x_test = test.iloc[:,0:8].copy()
y_test = test.iloc[:, 8].copy()

x_test,xm,xs = z_score_scaling(x_test)

predictions = model.predict(x_test)

r = predictions.shape[0]
accuracy = 0.0
for i in range(r):
    if abs(predictions[i]-y_test[i])<0.5: # do not change the tolerance as you'll be checked on +- 0.5 error only
        accuracy += 0.5
ok = 'Congratulations' if accuracy>95 else 'Optimization required'
print(f"{ok}, your accuracy is {accuracy}%")