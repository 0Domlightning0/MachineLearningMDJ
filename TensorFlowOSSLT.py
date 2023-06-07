import pandas as pd
import numpy as np

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
    "https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/Training_NoTitles.csv",
    names=['Enrolment','Latitude','Longitude','PercentageofStudentsWhoseFirstLanguageIsNotEnglish','PercentageofStudentsWhoseFirstLanguageIsNotFrench','PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry','PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry','PercentageofStudentsReceivingSpecialEducationServices','PercentageofStudentsIdentifiedasGifted','PercentageofGrade3StudentsAchievingtheProvincialStandardinReading','ChangeinGrade3ReadingAchievementOverThreeYears','PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting','ChangeinGrade3WritingAchievementOverThreeYears','PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematics','ChangeinGrade3MathematicsAchievementOverThreeYears','PercentageofGrade6StudentsAchievingtheProvincialStandardinReading','ChangeinGrade6ReadingAchievementOverThreeYears','PercentageofGrade6StudentsAchievingtheProvincialStandardinWriting','ChangeinGrade6WritingAchievementOverThreeYears','PercentageofGrade6StudentsAchievingtheProvincialStandardinMathematics','ChangeinGrade6MathematicsAchievementOverThreeYears'])


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

abalone_model.fit(abalone_features, abalone_labels, epochs=100)

normalize = layers.Normalization()

norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                           optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=100)


print(norm_abalone_model.predict([225,46.50593,-84.2873,0,100,0,5,15,0,69,-8,66,8,72,-9,80,0,80,13,-13]))

