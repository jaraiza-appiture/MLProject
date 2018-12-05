from sklearn import linear_model
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import shuffle
import sklearn
import pandas
pandas.set_option('display.max_rows', None)
RESULTS_PATH = '/home/jovan/Documents/CS475/MLProject/Results'

from AlloyLearning import AlloyDataPreper



class Stacker(object):
    """
    A transformer applying fitting a predictor `pred` to data in a way
        that will allow a higher-up predictor to build a model utilizing both this 
        and other predictors correctly.

    The fit_transform(self, x, y) of this class will create a column matrix, whose 
        each row contains the prediction of `pred` fitted on other rows than this one. 
        This allows a higher-level predictor to correctly fit a model on this, and other
        column matrices obtained from other lower-level predictors.

    The fit(self, x, y) and transform(self, x_) methods, will fit `pred` on all 
        of `x`, and transform the output of `x_` (which is either `x` or not) using the fitted 
        `pred`.

    Arguments:    
        pred: A lower-level predictor to stack.

        cv_fn: Function taking `x`, and returning a cross-validation object. In `fit_transform`
            th train and test indices of the object will be iterated over. For each iteration, `pred` will
            be fitted to the `x` and `y` with rows corresponding to the
            train indices, and the test indices of the output will be obtained
            by predicting on the corresponding indices of `x`.
    """
    def __init__(self, pred, cv_fn=lambda x: sklearn.cross_validation.KFold(x.shape[0],n_folds=10)):
        self._pred, self._cv_fn  = pred, cv_fn

    def fit_transform(self, x, y):
        x_trans = self._train_transform(x, y)

        self.fit(x, y)

        return x_trans

    def fit(self, x, y):
        """
        Same signature as any sklearn transformer.
        """
        self._pred.fit(x, y)

        return self

    def transform(self, x):
        """
        Same signature as any sklearn transformer.
        """
        return self._test_transform(x)

    def _train_transform(self, x, y):
        x_trans = np.nan * np.ones((x.shape[0], 1))

        all_te = set()
        for tr, te in self._cv_fn(x):
            all_te = all_te | set(te)
            x_trans[te, 0] = self._pred.fit(x[tr, :], y[tr]).predict(x[te, :]) 
        if all_te != set(range(x.shape[0])):
            warnings.warn('Not all indices covered by Stacker', sklearn.exceptions.FitFailedWarning)

        return x_trans

    def _test_transform(self, x):
        return self._pred.predict(x)

if __name__ == "__main__":

    #******************************9_12Cr****************************************
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
    dropna9_12Cr = ['CT Temp','CS','RT','AGS','AGS No.','EL','RA_2']

    #Features/Columns to remove from dataset
    exclude9_12Cr = ['MCR','0.5% CS','1.0% CS','2.0% CS','5.0% CS',
                    'UTS','Elong',
                    'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                    'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']

    # New Exclude based on Backward Selection
    dropna9_12Cr_reduced = ['CT Temp','CS','RT','RA_2']

    exclude9_12Cr_reduced = ['MCR','0.5% CS','1.0% CS','2.0% CS','5.0% CS',
                             'UTS','Elong',
                             'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                             'Temper3','ID','Hf','Homo','Re','Ta','Ti','O',
                             'P','AGS No.','Ni','EL','AGS','Nb'] #new drops
    dropna9_12Cr_reduced2 = ['CT Temp','CS','RT','RA_2','EL']

    exclude9_12Cr_reduced2 = ['MCR','0.5% CS','1.0% CS','2.0% CS','5.0% CS',
                              'UTS','Elong',
                              'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                              'Temper3','ID','Hf','Homo','Re','Ta','Ti','O',
                              'Fe', 'C', 'Cr', 'Mn', 'Si', 'Ni', 'Co', 'Mo', 'W', 'Nb', 'Al', 'P',
                              'Cu', 'V', 'B', 'N', 'S', 'Normal', 'Temper1', 'Temper2', 'AGS No.',
                              'AGS'] #new drops



    N9_12Cr = AlloyDataPreper(Dataset='9_12_Cr.csv',#name of dataset must match name of csv file located in RESULTS_PATH
                            label='RT',
                            dropna_cols=dropna9_12Cr_reduced,
                            exclude_cols=exclude9_12Cr_reduced,
                            fill_vals=fillvals,
                            )


    # y = np.random.randn(100)
    # x0 = (y + 0.1 * np.random.randn(100)).reshape((100, 1))
    # x1 = (y + 0.1 * np.random.randn(100)).reshape((100, 1))
    # x = np.zeros((100, 2))


    # g = ensemble.GradientBoostingRegressor()
    # l = linear_model.LinearRegression()

    # x[: 80, 0] = Stacker(g).fit_transform(x0[: 80, :], y[: 80])[:, 0]
    # x[: 80, 1] = Stacker(l).fit_transform(x1[: 80, :], y[: 80])[:, 0]

    # u = linear_model.LinearRegression().fit(x[: 80, :], y[: 80])

    # x[80: , 0] = Stacker(g).fit(x0[: 80, :], y[: 80]).transform(x0[80:, :])
    # x[80: , 1] = Stacker(l).fit(x1[: 80, :], y[: 80]).transform(x1[80:, :])
    # print(metrics.r2_score(y[80: ],u.predict(x[80:, :])))

    ###################################################################################
    Data = N9_12Cr.prep_it_split()
    Data['preds']['RT'] = Data['labels']
    TrainP,TestP = train_test_split(Data['preds'],test_size=0.20)
    TrainL = TrainP['RT'].values
    TrainP = TrainP.drop(['RT'],axis=1).values
    TestL = TestP['RT'].values
    TestP = TestP.drop(['RT'],axis=1).values

    # print("Train Preds:")
    # print(TrainP.head(100))
    # print("Train Labels:")
    # print(TrainL.head(100))

    scaler = StandardScaler()
    scaler.fit(TrainP)
    TrainP = scaler.transform(TrainP)
    TestP = scaler.transform(TestP)

    x = np.zeros((1405, 3))
    x2 = np.zeros((352, 3))

    r = ensemble.RandomForestRegressor(n_estimators=300,n_jobs=-1)
    g = ensemble.GradientBoostingRegressor(learning_rate=0.20, max_depth=5, n_estimators= 325)
    # r = MLPRegressor(max_iter=400,alpha=0.01,solver='lbfgs',activation='relu')
    # g = MLPRegressor(max_iter=400,alpha=0.01,solver='lbfgs',activation='relu')
    nn = MLPRegressor(max_iter=300,alpha=0.01,solver='lbfgs',activation='relu')

    # r_predictions_train = Stacker(r).fit_transform(TrainP,TrainL)[:,0]
    x[:,0] = Stacker(r).fit_transform(TrainP,TrainL)[:,0]

    x[:,1] = Stacker(g).fit_transform(TrainP,TrainL)[:,0]
    # nn_predictions_train = Stacker(nn).fit_transform(TrainP,TrainL)[:,0]
    x[:,2] = Stacker(nn).fit_transform(TrainP,TrainL)[:,0]

    u = linear_model.LinearRegression().fit(x[:,:],TrainL)

    # r_predictions_test = Stacker(r).fit(TrainP,TrainL).transform(TestP)
    x2[:,0] = Stacker(r).fit(TrainP,TrainL).transform(TestP)

    x2[:,1] = Stacker(g).fit(TrainP,TrainL).transform(TestP)

    # nn_predictions_test = Stacker(nn).fit(TrainP,TrainL).transform(TestP)
    x2[:,2] = Stacker(nn).fit(TrainP,TrainL).transform(TestP)

    print(metrics.r2_score(TestL,u.predict(x2[:,:])))

