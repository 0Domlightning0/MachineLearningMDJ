import pandas as pd
import numpy as np
from sklearn import linear_model

#py pip install -U scikit-learn

df = pd.read_csv('https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/TestCSV.csv')
df

reg = linear_model.LinearRegression()
reg.fit(df[["Start", "Double", "Triple"]].values, df.Final)

print(reg.predict([ [320,640,960] ]))

print(reg.score(df[["Start", "Double", "Triple"]].values, df.Final))
