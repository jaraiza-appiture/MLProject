# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from AlloyLearning import AlloyDataPreper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Extra imports necessary for the code

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X





####################################################################################################################################
#default values to fill in missiing values for the following features
fillvals = {'Fe': 0,'C':0,'Cr':0,'Mn':0,
            'Si':0,'Ni':0,'Co':0,'Mo':0,
            'W':0,'Nb':0,'Al':0,'P':0,
            'Cu':0,'Ti':0,'Ta':0,'Hf':0,
            'Re':0,'V':0,'B':0,'N':0,
            'O':0,'S':0,'Homo':0,'Normal':25,
            'Temper1':25,'Temper2':25,'Temper3':25}

#Drop any rows where a missing value in the following features exist
#even if just one of the following features is missing a value, entire row(instance/datapoint) will be dropped
dropna9_12Cr = ['CT Temp','CS','RT','AGS','AGS No.']

#Features/Columns to remove from dataset
exclude9_12Cr = ['UTS','Elong',
                  'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                  'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']#'B','Co','Temper2','Temper1']

exclude9_12Cr_2 = ['UTS','Elong','Normal','Fe','Cr','N','AGS No.','V',
                  'Mn','C','B','RA_2','Temper1','P','Si','Ni','Nb','1.0% CS','S','Mo',
                  'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                  'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']#'B','Co','Temper2','Temper1']


N9_12Cr = AlloyDataPreper(Dataset='9_12_Cr.csv',#name of dataset must match name of csv file located in RESULTS_PATH
                         label='RT',
                         dropna_cols=dropna9_12Cr,
                         exclude_cols=exclude9_12Cr_2,
                         fill_vals=fillvals,
                         )
ready9_12Cr = N9_12Cr.prep_it_split()


transformer = ReduceVIF()

# Only use 10 columns for speed in this example
X = ready9_12Cr['preds']
y = ready9_12Cr['labels']
X = transformer.fit_transform(X)

X.head()
