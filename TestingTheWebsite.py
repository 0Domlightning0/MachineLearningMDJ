import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

abalone_train = pd.read_csv(
    "https://wellingtoncdsbca-my.sharepoint.com/personal/wr648190_wellingtoncdsb_ca/Documents/Names_missing_OSSLT.csv?web=1",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
print(abalone_features)

