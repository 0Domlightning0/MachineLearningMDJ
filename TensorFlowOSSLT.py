import pandas as pd
import numpy as np
import math

# python -m pip install pandas
# python -m pip install numpy
# python -m pip install openpyxl

# Make numpy values easier to read.
np.set_printoptions(precision=10, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=10, suppress=True)

abalone_train = pd.read_csv(
    "https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/OnlyGr6MathCSVnoNames.csv",
    names=['Enrolment','Latitude','Longitude','PercentageofStudentsWhoseFirstLanguageIsNotEnglish','PercentageofStudentsWhoseFirstLanguageIsNotFrench','PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry','PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry','PercentageofStudentsReceivingSpecialEducationServices','PercentageofStudentsIdentifiedasGifted','PercentageofGrade3StudentsAchievingtheProvincialStandardinReading','ChangeinGrade3ReadingAchievementOverThreeYears','PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting','ChangeinGrade3WritingAchievementOverThreeYears','PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematics','ChangeinGrade3MathematicsAchievementOverThreeYears','PercentageofGrade6StudentsAchievingtheProvincialStandardinMathematics'])


abalone_train.head()

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('PercentageofGrade6StudentsAchievingtheProvincialStandardinMathematics')

abalone_features = np.array(abalone_features)
abalone_features

abalone_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=1000)

normalize = layers.Normalization()

norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])


norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                           optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=1000)




#Prediction


#Values of the CSV file in a list format for easy processing
thingy = pd.read_csv("https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/OnlyGr6MathCSVnoNames.csv", nrows=0,skiprows=4,header=0)

# Index out the brackets
thingy = str(thingy)
x = thingy.index(']')
bozo = (str(thingy)[26:(x)])

# Takes the values and turns it into an integer list
bozo = bozo.split(', ')
res = [eval(i) for i in bozo]


print(res[15])

print(norm_abalone_model.predict([res[0:15]]))

Total_correct = 0

Total_predict = 0

Total_incorrect = 0
for i in range(100):
    pd.read_csv("https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/OnlyGr6MathCSVnoNames.csv", nrows=0,skiprows=i,header=0)
    Total_incorrect += abs(res[15] - norm_abalone_model.predict([res[0:15]]))

print(Total_incorrect/100)
    

