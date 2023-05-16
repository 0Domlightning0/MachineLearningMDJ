import pandas as pd
import numpy as np
from sklearn import linear_model

#py pip install -U scikit-learn

df = pd.read_csv('https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/OSSLT_Train.csv')
df



#Removing All NaN Values within the CSV with the median value so it doesnt give an error while computing the weights 


df.PercentageofGrade3StudentsAchievingtheProvincialStandardinReading.median()

df.PercentageofGrade3StudentsAchievingtheProvincialStandardinReading = df.PercentageofGrade3StudentsAchievingtheProvincialStandardinReading.fillna(df.PercentageofGrade3StudentsAchievingtheProvincialStandardinReading.median())
df


df.Longitude.median()

df.Longitude = df.Longitude.fillna(df.Longitude.median())
df

df.PercentageofStudentsWhoseFirstLanguageIsNotEnglish.median()

df.PercentageofStudentsWhoseFirstLanguageIsNotEnglish = df.PercentageofStudentsWhoseFirstLanguageIsNotEnglish.fillna(df.PercentageofStudentsWhoseFirstLanguageIsNotEnglish.median())
df

df.PercentageofStudentsWhoseFirstLanguageIsNotFrench.median()
df.PercentageofStudentsWhoseFirstLanguageIsNotFrench = df.PercentageofStudentsWhoseFirstLanguageIsNotFrench.fillna(df.PercentageofStudentsWhoseFirstLanguageIsNotFrench.median())
df

df.PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry.median()
df.PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry = df.PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry.fillna(df.PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry.median())
df

df.PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry.median()
df.PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry = df.PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry.fillna(df.PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry.median())
df

df.PercentageofStudentsReceivingSpecialEducationServices.median()
df.PercentageofStudentsReceivingSpecialEducationServices = df.PercentageofStudentsReceivingSpecialEducationServices.fillna(df.PercentageofStudentsReceivingSpecialEducationServices.median())
df

df.PercentageofStudentsIdentifiedasGifted.median()
df.PercentageofStudentsIdentifiedasGifted	 = df.PercentageofStudentsIdentifiedasGifted	.fillna(df.PercentageofStudentsIdentifiedasGifted	.median())
df

df.ChangeinGrade3ReadingAchievementOverThreeYears.median()
df.ChangeinGrade3ReadingAchievementOverThreeYears = df.ChangeinGrade3ReadingAchievementOverThreeYears.fillna(df.ChangeinGrade3ReadingAchievementOverThreeYears.median())
df

df.PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting.median()
df.PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting = df.PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting.fillna(df.PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting.median())
df











# Creation of Linear weights and intercept with variale names in order 
reg = linear_model.LinearRegression()
reg.fit(df[['Enrolment',"Latitude","Longitude","PercentageofStudentsWhoseFirstLanguageIsNotEnglish","PercentageofStudentsWhoseFirstLanguageIsNotFrench","PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry","PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry","PercentageofStudentsReceivingSpecialEducationServices","PercentageofStudentsIdentifiedasGifted","PercentageofGrade3StudentsAchievingtheProvincialStandardinReading","ChangeinGrade3ReadingAchievementOverThreeYears","PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting","ChangeinGrade3WritingAchievementOverThreeYears","PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematics","ChangeinGrade3MathematicsAchievementOverThreeYears",]].values, df.PercentageofGrade6StudentsAchievingtheProvincialStandardinMathematics)



#Values of the CSV file in a list format for easy processing
thingy = pd.read_csv("https://raw.githubusercontent.com/0Domlightning0/MachineLearningOffical/main/OSSLT_Eval.csv", nrows=0,skiprows=1,header=0)

thingy = str(thingy)
x = thingy.index(']')

bozo = (str(thingy)[26:(x)])
bozo = bozo.split(', ')
res = [eval(i) for i in bozo]



# Assigning the Variable input with its matching value in the list
Enrolment_Input = float(res[0])

Latitude_Input = float(res[1])

LongitudeInput = float(res[2])

PercentageofStudentsWhoseFirstLanguageIsNotEnglishInput = float(res[3])

PercentageofStudentsWhoseFirstLanguageIsNotFrenchInput = float(res[4])

PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountryInput = float(res[5])

PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountryInput = float(res[6])

PercentageofStudentsReceivingSpecialEducationServicesInput = float(res[7])

PercentageofStudentsIdentifiedasGiftedInput = float(res[8])

PercentageofGrade3StudentsAchievingtheProvincialStandardinReadingInput = float(res[9])

ChangeinGrade3ReadingAchievementOverThreeYearsInput = float(res[10])

PercentageofGrade3StudentsAchievingtheProvincialStandardinWritingInput = float(res[11])

ChangeinGrade3WritingAchievementOverThreeYearsInput = float(res[12])

PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematicsInput = float(res[13])

ChangeinGrade3MathematicsAchievementOverThreeYearsInput = float(res[14])



#Prints  out the bais/y int of the linear equasion  
print(reg.intercept_)
print()

#Prints  out the Input/Actual Value mulitplied by its given weight 
print( reg.coef_[0] * Enrolment_Input)
print( reg.coef_[1] * Latitude_Input)
print( LongitudeInput * reg.coef_[2])
print( PercentageofStudentsWhoseFirstLanguageIsNotEnglishInput * reg.coef_[3])
print(PercentageofStudentsWhoseFirstLanguageIsNotFrenchInput * reg.coef_[4])
print(PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountryInput + reg.coef_[5] )
print(PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountryInput * reg.coef_[6])
print(PercentageofStudentsReceivingSpecialEducationServicesInput * reg.coef_[7])
print(PercentageofStudentsIdentifiedasGiftedInput * reg.coef_[8] )
print(PercentageofGrade3StudentsAchievingtheProvincialStandardinReadingInput * reg.coef_[9])
print(ChangeinGrade3ReadingAchievementOverThreeYearsInput * reg.coef_[10])
print(PercentageofGrade3StudentsAchievingtheProvincialStandardinWritingInput * reg.coef_[11])
print(ChangeinGrade3WritingAchievementOverThreeYearsInput * reg.coef_[12])
print(PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematicsInput * reg.coef_[13])
print(ChangeinGrade3MathematicsAchievementOverThreeYearsInput * reg.coef_[14])


print()

#Predicts the % of grade 6's getting lvl 3's in Math
print(reg.predict([ res[0:15] ]))


# Manual version of the predict function 
#print(reg.intercept_ + reg.coef_[0] * Enrolment_Input + reg.coef_[1] * Latitude_Input + LongitudeInput * reg.coef_[2] + PercentageofStudentsWhoseFirstLanguageIsNotEnglishInput * reg.coef_[3] + PercentageofStudentsWhoseFirstLanguageIsNotFrenchInput * reg.coef_[4] + PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountryInput * reg.coef_[5] + PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountryInput * reg.coef_[6] + PercentageofStudentsReceivingSpecialEducationServicesInput * reg.coef_[7] + PercentageofStudentsIdentifiedasGiftedInput * reg.coef_[8] +PercentageofGrade3StudentsAchievingtheProvincialStandardinReadingInput* reg.coef_[9] + ChangeinGrade3ReadingAchievementOverThreeYearsInput * reg.coef_[10] + PercentageofGrade3StudentsAchievingtheProvincialStandardinWritingInput * reg.coef_[11] + ChangeinGrade3WritingAchievementOverThreeYearsInput * reg.coef_[12] + PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematicsInput * reg.coef_[13] + ChangeinGrade3MathematicsAchievementOverThreeYearsInput * reg.coef_[14])

print("Actual answer:")

# Takes the value traight from the CSV to show the real answer
print(res[19])

#Calculates the score of the model Best score is 1.0
print(reg.score(df[['Enrolment',"Latitude","Longitude","PercentageofStudentsWhoseFirstLanguageIsNotEnglish","PercentageofStudentsWhoseFirstLanguageIsNotFrench","PercentageofStudentsWhoAreNewtoCanadafromaNonEnglishSpeakingCountry","PercentageofStudentsWhoAreNewtoCanadafromaNonFrenchSpeakingCountry","PercentageofStudentsReceivingSpecialEducationServices","PercentageofStudentsIdentifiedasGifted","PercentageofGrade3StudentsAchievingtheProvincialStandardinReading","ChangeinGrade3ReadingAchievementOverThreeYears","PercentageofGrade3StudentsAchievingtheProvincialStandardinWriting","ChangeinGrade3WritingAchievementOverThreeYears","PercentageofGrade3StudentsAchievingtheProvincialStandardinMathematics","ChangeinGrade3MathematicsAchievementOverThreeYears",]].values , df.PercentageofGrade6StudentsAchievingtheProvincialStandardinMathematics  ))
