import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

Student_train = pd.read_csv(
    "https://wellingtoncdsbca-my.sharepoint.com/personal/wr648190_wellingtoncdsb_ca/Documents/No_missing_OSSLT.csv?web=1",
    names=["Enrolment","Latitude","Longitude","Percentage of Students Whose First Language Is Not English","Percentage of Students Whose First Language Is Not French","Percentage of Students Who Are New to Canada from a Non-English Speaking Country","Percentage of Students Who Are New to Canada from a Non-French Speaking Country","Percentage of Students Receiving Special Education Services","Percentage of Students Identified as Gifted","Percentage of Grade 3 Students Achieving the Provincial Standard in Reading","Change in Grade 3 Reading Achievement Over Three Years","Percentage of Grade 3 Students Achieving the Provincial Standard in Writing","Change in Grade 3 Writing Achievement Over Three Years","Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics"	"Change in Grade 3 Mathematics Achievement Over Three Years","Percentage of Grade 6 Students Achieving the Provincial Standard in Reading", "Change in Grade 6 Reading Achievement Over Three Years"	"Percentage of Grade 6 Students Achieving the Provincial Standard in Writing"	"Change in Grade 6 Writing Achievement Over Three Years","Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics","Change in Grade 6 Mathematics Achievement Over Three Years"])


Student_train.head()

Student_features = Student_train.copy()
Student_labels = Student_features.pop("Percentage of Grade 3 Students Achieving the Provincial Standard in Reading")
Student_features = np.array(Student_features)
Student_features
