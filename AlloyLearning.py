#Programmer: Jovan Araiza

#%%
# Math Modules
import numpy as np
import pandas as pd
from scipy import stats
# Utility Modules
import os
from ast import literal_eval
from operator import itemgetter
# Machine Learning Modules
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, LeaveOneOut, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ARDRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.utils import shuffle
# Graphing Modules

import matplotlib
import matplotlib.pyplot as plt
np.random.seed(42)

# To plot pretty figures
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the results | Change this to your directory location
RESULTS_PATH = '/home/jovan/Documents/CS475/MLProject/Results'


#Class Definitions
class AlloyDataPreper():
    """
    DESCRIPTION: Loads CSV file, performs some cleaning routines, seperates labels and predictors.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        Dataset : str
            Name of CSV file to load
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        label : str
            Name of label i.e. RT (Rupture Time)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dropna_cols : list of str
            List of column names. Rows with missing values in these columns will be dropped.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        exclude_cols : list of str
            List of column names. These columns will be dropped from the dataset.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fill_vals : dict of {str : int/float} pairs
            Dictionary of column names to corresponding fill values.
            Replace any empty/Nan values within column/feature N with corresponding fill value for N.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        strat : bool
            If true, will perform stratified sampling based on column name(strat_col) provided.
            Must provide strat_col.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        start_col : str
            Name of column to stratify by.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        strat_lim : int
            Limit how many strata to have. Higher values means less strata.

    """

    def __init__(self,Dataset='',
                      label='',
                      dropna_cols=[],
                      exclude_cols=[],
                      fill_vals=dict(),
                      strat=False,
                      strat_col='',
                      strat_lim=1):
        '''[summary]

        Keyword Arguments:
            Dataset {str} -- [description] (default: {''})
            label {str} -- [description] (default: {''})
            dropna_cols {list} -- [description] (default: {[]})
            exclude_cols {list} -- [description] (default: {[]})
            fill_vals {[type]} -- [description] (default: {dict()})
            strat {bool} -- [description] (default: {False})
            strat_col {str} -- [description] (default: {''})
            strat_lim {int} -- [description] (default: {1})
        '''

        self.__Dataset = Dataset# Dataset name str
        self.__Alloy_Data = dict()# Keeps Original Data set
        self.__Ready_Data = dict()
        self.__fill_values = fill_vals
        self.__strat = strat
        self.__strat_lim = strat_lim
        self.__strat_col = strat_col
        self.__label = label
        self.__exclude_cols = exclude_cols
        self.__dropna_cols = dropna_cols

    def set_strat_col(self,strat_col):
        '''
        Set column name used to stratify data.
        '''
        self.__strat_col = strat_col

    def set_strat_lim(self,strat_lim):
        '''
        Set strata limit. Higher value means less strata.
        '''
        self.__strat_lim = strat_lim

    def __load_alloy_data(self,alloy_path=RESULTS_PATH,file='Data.csv'):
        '''
        Load CSV file and convert to DataFrame.
        '''
        csv_path = os.path.join(alloy_path, file)
        return pd.read_csv(csv_path)

    def __clean_alloy_data(self,Alloy_Data):
        """
        DESCRIPTION: Drops all rows with no data at all.
        Drops all rows with missing data from columns in dropna_cols list.
        Replaces missing values for columns with corresponding fill values.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PARAMETERS:
            Alloy_Data : DataFrame
                Original data loaded.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        RETURNS:
            DataFrame
                Cleaned data.
        """
        #Drop all rows with no data at all
        Alloy_Data = Alloy_Data.dropna(how='all')
        Alloy_Data = Alloy_Data.reset_index(drop=True)

        #Drop all rows with missing data from columns in dropna_cols list
        Alloy_Data = Alloy_Data.dropna(subset=self.__dropna_cols)
        Alloy_Data = Alloy_Data.reset_index(drop=True)

        #Replace missing values for columns with corresponding fill values
        Alloy_Data = Alloy_Data.fillna(value=self.__fill_values)

        return Alloy_Data

    def __stratify_split(self,Alloy_Data):
        """
        !!!Disabled/Not Used!!!

        DESCRIPTION: Performs stratified sampling based on strat_col and strat_lim.
        Splits data into train and test sets.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PARAMETERS:
            Alloy_Data : DataFrame
                Original data loaded.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        RETURNS:
            tuple of DataFrames (DataFrame,DataFrame)
                train set and test set

        """
        CAlloy_Data = Alloy_Data.copy()
        #Column to Add
        StratCol = self.__strat_col+'_cat'
        #Add Strat Column
        CAlloy_Data[StratCol] = np.ceil(CAlloy_Data[self.__strat_col] / self.__strat_lim)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        #Split
        for train_index, test_index in split.split(CAlloy_Data, CAlloy_Data[StratCol]):
            strat_train_set = CAlloy_Data.loc[train_index]
            strat_test_set = CAlloy_Data.loc[test_index]
        #remove Strat Column
        for set_ in (strat_train_set, strat_test_set):
            set_.drop(StratCol,axis=1,inplace=True)

        return strat_train_set,strat_test_set

    def __load_clean_data(self):
        '''
        Loads and cleans dataset
        '''
        NewD = self.__load_alloy_data(file=self.__Dataset)
        NewD = self.__clean_alloy_data(NewD)
        self.__Alloy_Data.update({self.__Dataset:NewD})

    def __drop(self,Alloy_Data):
        '''[summary]
        Arguments:
            Alloy_Data {[type]} -- [description]
        Returns:
            [type] -- [description]
        '''
        Alloy_Data = Alloy_Data.drop(self.__exclude_cols,axis=1)
        return Alloy_Data

    def __split(self,Alloy_Data):
        '''
        Splits predictors and labels
        '''
        #drop exclusion columns
        Alloy_Data = self.__drop(Alloy_Data)

        # Seperate Labels and Predictors
        preds = Alloy_Data.drop(self.__label,axis=1)#drop label column for training set
        labels = Alloy_Data[self.__label].copy()

        return preds,labels

    def __shuffleData(self):
        '''
        Shuffles dataset, splits predictors and labels
        '''
        for DName,DSet in self.__Alloy_Data.items():
            CDset = DSet.copy()
            shuffledDset = shuffle(CDset)
            #preds,labels = self.__split(shuffledDset)
            shuffledDset = self.__drop(shuffledDset)
            self.__Ready_Data = {'data':shuffledDset,'name':DName}

    def __shuffleData_split(self):
        '''
        Shuffles dataset, splits predictors and labels
        '''
        for DName,DSet in self.__Alloy_Data.items():
            CDset = DSet.copy()
            shuffledDset = shuffle(CDset)
            preds,labels = self.__split(shuffledDset)
            self.__Ready_Data = {'preds':preds,'labels':labels,'name':DName}

    def prep_it(self):

        self.__load_clean_data()
        self.__shuffleData()
        return self.__Ready_Data
    def prep_it_split(self):
        '''
        Main Funciton

        DESCRIPTION: Load, clean, shuffle, and split data into predictors and labels
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        RETURNS:
            dict {'preds':predictors,'labels':labels}

                -predictors are contained in a pandas DataFrame

                -labels are contained in a numpy Series
        '''
        self.__load_clean_data()
        self.__shuffleData_split()
        return self.__Ready_Data

    def log_it(self,preds=[],label=False):
        '''
        DESCRIPTION: Apply log to any column from predictors. Apply log to label.
        Changes column names from 'colname' to 'log_colname'.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PARAMETERS:
            preds : list of str
                List of predictor column names that will be converted to log
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            label : bool
                If True, will apply log to labels
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        RETURNS:
            dict {'preds':predictors,'labels':labels}

                -predictors are contained in a DataFrame

                -labels are contained in a Series
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PRECONDITIONS:
            Must have executed prep_it() first
        '''
        #if data has been preped
        if len(preds) == 0 and ~label:
            return self.__Ready_Data
        if len(self.__Ready_Data) == 0:
            self.prep_it()

        log_Ready_Data= {'preds':None,'labels':None,'name':self.__Ready_Data['name']}

        log_Ready_Data['preds'] = self.__Ready_Data['preds'].copy()

        if len(preds):
            for pred in preds:
                pred_log = 'log_'+pred

                pr = log_Ready_Data['preds']
                pr[pred_log]= log_Ready_Data['preds'][pred].apply(lambda x : np.log(x))
                pr=pr.drop([pred],axis=1)
                log_Ready_Data['preds'] = pr

        log_Ready_Data['labels'] = self.__Ready_Data['labels'].copy()

        if label:
            label_log = 'log_'+self.__label

            ll=pd.DataFrame({self.__label:log_Ready_Data['labels'].values})
            ll[label_log] = ll[self.__label].apply(lambda x : np.log(x))
            ll.drop([self.__label],axis=1,inplace=True)
            #keep same index vals
            dic_ = dict(list(zip(ll[label_log].index,log_Ready_Data['labels'].index)))
            log_Ready_Data['labels']=ll[label_log].rename(dic_)

        return log_Ready_Data

class myGSCV():
    """
    DESCRIPTION: Performs grid search with cross validation on a given model(estimator) to
    find best combination of hyperparamters for that model.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        estimator : ML algorithm (must have functions fit() and predict())
            Estimator is the ML algorithm to be tested.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model_param_grid : dict
            A dictionary containing parameter values to try for parameters of model.

            RandomForestRegressor EX: mpg = {'n_estimators':[100,200,300],'max_features':[10,15,20]}
            Will instantiate model with following combinations : n_estimators=100,max_features=10
                                                                 n_estimators=100,max_features=15
                                                                                 .
                                                                                 .
                                                                                 .
                                                                 n_estimators=300,max_features=20
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tran_param_grid : dict
            A dictionary containing preprocessing values to try.

            EX: tpg = {'tr_scaler':[StandardScaler,MinMaxScaler]}
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        gscv : int [default=10]
            Number of folds to perform (KFold).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        label : str
            Name of label.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        file_ : file
            file to write results to.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        apply : bool [default=False]
            If True, will transform predictons mabe by model and corresponding labels with applyFunc provided.
            ApplyFunc must be provided!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        applyFunc : function [default=None]
            Function to be applied to predictions made by model and corresponding labels.
    """
    def __init__(self,estimator,label,model_param_grid,tran_param_grid,file_,apply=False,applyFunc=None,gscv=10):
        self.__estimator = estimator
        self.__model_param_grid = model_param_grid
        self.__tran_param_grid = tran_param_grid
        self.__gscv = gscv
        self.__label = label
        self.best_combo = None# Contains best model and preprocessing param combination
        self.best_combo_results = None # Stats for best combination
        self.results = [] #Contains stats for all combinations
        self.__num_combos = 0
        self.__combineparams()
        self.__file_ = file_
        self.__apply = apply
        self.__applyFunc = applyFunc #When defining applyFunc, applyData will be passed to applyFunc in second parameter.

    def __split(self,data):
        '''
        Splits predictors and labels
        '''
        # Seperate Labels and Predictors
        preds = data.drop(self.__label,axis=1)
        labels = data[self.__label].copy()
        return preds,labels

    def __combineparams(self):
        '''
        Combine model parameter grid with preprocessing grid.

        This will allow all possible combinations of
        model and preprocessing hyperparameters to be created.
        '''
        for dic in self.__tran_param_grid:
            dic.update(self.__model_param_grid)

    def __split_combo(self,combos):
        '''
        DESCRIPTION: Separates model and preprocessing combinations.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PARAMETERS:
            combos : dict
                Contains all possible combinations of model params and preprocessing params
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        RETURNS:
            list of tuples : [(model_hyperparams1,preprocessing_params1),...,(model_hyperparamsN,preprocessing_paramsN)]
                Contains a list of all possible model and preprocessing params combinations separated in a tuple.
        '''
        preprocessing_params = ['tr_scaler',]# Can add more preprocessing params to separate
        #                       'pca_whiten',
        #                       'pca_kernel',
        #                       'pca_gamma',
        #                       'pca_alpha',
        #                       'pca_ridge']
        new_l = []
        s_p = dict()
        h_c = dict()
        for dic in combos:
            s_p.clear()
            h_c.clear()
            for key,item in dic.items():
                #Split them here
                if key in preprocessing_params:
                    s_p.update({key:item})
                else:
                    h_c.update({key:item})
            new_l.append((h_c.copy(),s_p.copy()))
        return new_l

    def optimize(self,data,applyData=None):
        '''
        Main Function

        DESCRIPTION: Performs grid search with cross validation on given data set. Finds best
        combination of model and preprocessing params with smallest test rmse score.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PARAMETERS:

            data : DataFrame
                Data to test model on.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            applyData : None [Default=None]
                applyData can be any number of types depending on given applyFunc.
                When defining applyFunc, applyData will be passed to applyFunc in second parameter.
                This is meant for applying specific transformations to data after predictions are made.

        '''
        fold = KFold(n_splits=self.__gscv)

        # generate all possible combinations of hyperparams
        combinations = list(ParameterGrid(self.__tran_param_grid))
        combinations = self.__split_combo(combinations)

        self.__num_combos = len(combinations)

        print('Combinations to try:',self.__num_combos)
        self.results.clear()
        print('Innner Cross Val:')

        for hy_combo,tr_combo in combinations:
            # prepare results for combination
            self.results.append(({'hyper_params':hy_combo,'transform_params':tr_combo},dict()))

            print('\nModel Params: %s'%(hy_combo))
            print('Transform Params: %s'%(tr_combo))

            # instantiate estimator with desired hyperparams
            model = self.__estimator(**hy_combo)

            test_scores = {'rmse':[],'mean_rmse':None,'std_rmse':None}
            train_scores = {'rmse':[],'mean_rmse':None,'std_rmse':None}

            # perform cross validation with KFold
            for train_in,test_in in fold.split(data):
                # split test and train data
                train_data = data.iloc[train_in]
                test_data = data.iloc[test_in]

                if self.__apply:
                    train_applyData = applyData.iloc[train_in]
                    test_applyData = applyData.iloc[test_in]

                # split predictors and labels
                train_predictors,train_labels = self.__split(train_data)
                test_predictors,test_labels = self.__split(test_data)

                # scale
                scaler = tr_combo['tr_scaler']()
                scaler.fit(train_predictors)
                train_predictors = scaler.transform(train_predictors)
                test_predictors = scaler.transform(test_predictors)
        #                 # reduce dimensions w/ PCA
        #                 if tr_combo['PCA'] == 'PCA':
        #                     pca = PCA(n_components=tr_combo['pca_n_components'],
        #                               whiten=tr_combo['pca_whiten'])
        #                     pca.fit(train_predictors)
        #                     train_predictors = pca.transform(train_predictors)
        #                     test_predictors = pca.transform(test_predictors)
        #                 elif tr_combo['PCA'] == 'KernelPCA':
        #                     pca = KernelPCA(n_components=tr_combo['pca_n_components'],
        #                                     kernel='poly')#tr_combo['pca_kernel'],
        #                                     #gamma=tr_combo['pca_gamma'],
        #                                     #n_jobs=-1)
        #                     pca.fit(train_predictors)
        #                     train_predictors = pca.transform(train_predictors)
        #                     test_predictors = pca.transform(test_predictors)
        #                 elif tr_combo['PCA'] == 'SparsePCA':
        #                     pca = SparsePCA(n_components=tr_combo['pca_n_components'])
        #                                     #alpha=tr_combo['pca_alpha'],
        #                                     #ridge_alpha=tr_combo['pca_ridge'],
        #                                     #n_jobs=-1)
        #                     pca.fit(train_predictors)
        #                     train_predictors = pca.transform(train_predictors)
        #                     test_predictors = pca.transform(test_predictors)

                # train model
                model.fit(train_predictors,train_labels)
                # make predictions
                test_predictions = model.predict(test_predictors)
                train_predictions = model.predict(train_predictors)

                #apply more transformations
                if self.__apply:
                    train_predictions = self.__applyFunc(train_predictions,train_applyData.copy())
                    test_predictions = self.__applyFunc(test_predictions,test_applyData.copy())
                    train_labels = self.__applyFunc(train_labels,train_applyData.copy())
                    test_labels = self.__applyFunc(test_labels,test_applyData.copy())

                # performance measure
                test_rmsef = rmsef(test_predictions,test_labels)
                train_rmsef = rmsef(train_predictions,train_labels)
                print('Test RMSE:',test_rmsef)
                print('Train RMSE:',train_rmsef)
                test_scores['rmse'].append(test_rmsef)
                train_scores['rmse'].append(train_rmsef)

            test_scores['mean_rmse'] = np.mean(test_scores['rmse'])
            train_scores['mean_rmse'] = np.mean(train_scores['rmse'])

            test_scores['std_rmse'] = np.std(test_scores['rmse'])
            train_scores['std_rmse'] = np.std(train_scores['rmse'])

            # Append results of combination
            self.results[-1][-1].update({'train_scores':train_scores,'test_scores':test_scores})

        #Sort by mean test rmse score
        self.results = sorted(self.results,key=lambda x: x[-1]['test_scores']['mean_rmse'])

        #Save best combination
        self.best_combo = self.results[0][0]

        #Print results to terminal and text file
        print('\n\n(((((((((((((((((((((((((((((((((((((( Inner Cross Val Results ))))))))))))))))))))))))))))))))))))))')
        print('\n\n(((((((((((((((((((((((((((((((((((((( Inner Cross Val Results ))))))))))))))))))))))))))))))))))))))',file=self.__file_)
        if len(self.results) < 5:
            stopper = len(self.results)
        else:
            stopper = 5
            print('Top 5 Combinations')
            print('Top 5 Combinations',file=self.__file_)
        for best_c in self.results[:stopper]:
            print('\nParams:')
            print('\nParams:',file=self.__file_)
            print('\tModel: [%s]'%(best_c[0]['hyper_params']))
            print('\tModel: [%s]'%(best_c[0]['hyper_params']),file=self.__file_)
            print('\tPreprocessing: [%s]'%(best_c[0]['transform_params']))
            print('\tPreprocessing: [%s]'%(best_c[0]['transform_params']),file=self.__file_)
            print('Results:')
            print('Results:',file=self.__file_)
            print('\tTrain Score:')
            print('\tTrain Score:',file=self.__file_)
            print('\t\tMean RMSE: ',best_c[1]['train_scores']['mean_rmse'])
            print('\t\tMean RMSE: ',best_c[1]['train_scores']['mean_rmse'],file=self.__file_)
            print('\tTest Score:')
            print('\tTest Score:',file=self.__file_)
            print('\t\tMean RMSE: ',best_c[1]['test_scores']['mean_rmse'])
            print('\t\tMean RMSE: ',best_c[1]['test_scores']['mean_rmse'],file=self.__file_)
            print('************************************************************************')
            print('************************************************************************',file=self.__file_)
        #Save best combination stats
        self.best_combo_results = self.results[0][1]

class CrossVal():
    '''
    DESCRIPTION: Performs cross validation on model with an inner grid
    search/cross validation step to optimize model and preprocessing parameters
    before training and testing model in outer cross validation.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        estimator : ML algorithm (must have functions fit() and predict())
            Estimator is the ML algorithm to be tested. Any ML algorithm defined with fit() and predict() methods
            will work. predict() must return a numpy ndarray.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        data : DataFrame
            Combined predictors and labels data to test ML algorithm on.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        label : str
            Name of label.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model_param_grid : dict
            A dictionary containing parameter values to try for parameters of ML algorithm.

            RandomForestRegressor EX: mpg = {'n_estimators':[100,200,300],'max_features':[10,15,20]}
            Will instantiate model with following combinations : n_estimators=100,max_features=10
                                                                 n_estimators=100,max_features=15
                                                                                 .
                                                                                 .
                                                                                 .
                                                                 n_estimators=300,max_features=20
            ! Will be used in myGSCV class only for param optimization !
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tran_param_grid : dict
            A dictionary containing preprocessing values to try.

            EX: tpg = {'tr_scaler':[StandardScaler,MinMaxScaler]}
            Standard and MinMax scaler will be used to transform data in preprocessing.
            !Any scaler with fit() and transform() function will work!
            !Will be used in myGSCV class only for param optimization!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        file_ : file
            file to write results to.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        apply : bool [default=False]
            If True, will transform predictons mabe by model and corresponding labels with applyFunc provided.
            ApplyFunc must be provided!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        applyFunc : function [default=None]
            Function to be applied to predictions made by model and corresponding labels.
            Useful if applying log to labels, need to unlog the predictions and labels before
            calculating performance measures (RMSE and R-Squared).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        applyData : None [default=None]
            applyData can be any number of types depending on given applyFunc.

            !When defining applyFunc, applyData will be passed to applyFunc in second parameter!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cv : int [default=10]
            Number of outer folds to perform (KFold).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        gscv : int [default=10]
            Number of inner folds to perform (KFold).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        looCV : bool [default=False]
            If True, will perform Leave one out on outer cross validation only.

    '''
    def __init__(self,estimator,
                      data,
                      label,
                      model_param_grid,
                      tran_param_grid,
                      file_,
                      apply=False,
                      applyFunc=None,
                      applyData=None,
                      cv=10,
                      gscv=10,
                      looCV=False):
        '''[summary]

        Arguments:
            estimator {[type]} -- [description]
            data {[type]} -- [description]
            label {[type]} -- [description]
            model_param_grid {[type]} -- [description]
            tran_param_grid {[type]} -- [description]
            file_ {[type]} -- [description]

        Keyword Arguments:
            apply {bool} -- [description] (default: {False})
            applyFunc {[type]} -- [description] (default: {None})
            applyData {[type]} -- [description] (default: {None})
            cv {int} -- [description] (default: {10})
            gscv {int} -- [description] (default: {10})
            looCV {bool} -- [description] (default: {False})
        '''

        self.__estimator = estimator
        self.__data = data
        self.__label = label
        self.__model_param_grid = model_param_grid
        self.__tran_param_grid = tran_param_grid
        self.__cv = cv
        self.__looCV = looCV
        self.results = None
        self.combos = []
        self.__apply = apply
        self.__applyFunc = applyFunc
        self.__applyData = applyData
        self.__file_ = file_
        self.__paramOptimizer = myGSCV(estimator=self.__estimator,
                                     label=self.__label,
                                     model_param_grid=self.__model_param_grid,
                                     tran_param_grid=self.__tran_param_grid,
                                     file_=self.__file_,
                                     apply=self.__apply,
                                     applyFunc=self.__applyFunc,
                                     gscv=gscv)

    def __split(self,data):
        '''
        Separates labels and predictors
        '''
        # Seperate Labels and Predictors
        preds = data.drop(self.__label,axis=1) # drop label column for training set
        labels = data[self.__label].copy()

        return preds,labels

    def validate(self):
        '''
        DESCRIPTION: Performs cross validation on given model and
        dataset with optimized model and preprocessing params. Saves results to
        a text file.
        '''
        if self.__looCV:
            fold = LeaveOneOut()
        else:
            fold = KFold(n_splits=self.__cv)

        self.combos.clear()

        if self.__looCV or (len(self.__data)/self.__cv) < 10:
            test_scores = {'predictions':[],'labels':[],'rmse':None,'r_squared':None}
        else:
            test_scores = {'rmse':([],[]),'mean_rmse':None,'std_rmse':None,'r_squared':[],'mean_r_squared':None,'std_r_squared':None}

        train_scores = {'rmse':([],[]),'mean_rmse':None,'std_rmse':None,'r_squared':[],'mean_r_squared':None,'std_r_squared':None}

        for train_in,test_in in fold.split(self.__data):

            # split test and train data
            train_data = self.__data.iloc[train_in]
            test_data = self.__data.iloc[test_in]
            if self.__apply:
                train_applyData = self.__applyData.iloc[train_in]
                test_applyData = self.__applyData.iloc[test_in]

            # split predictors and labels
            train_predictors,train_labels = self.__split(train_data)
            test_predictors,test_labels = self.__split(test_data)

            # optimize model hyperparams
            if self.__apply:
                self.__paramOptimizer.optimize(train_data,train_applyData)
            else:
                self.__paramOptimizer.optimize(train_data)
            hy_params = self.__paramOptimizer.best_combo['hyper_params']
            tr_params = self.__paramOptimizer.best_combo['transform_params']
            params = {}
            params.update(hy_params)
            params.update(tr_params)
            print('Optimized Model params:',hy_params)
            print('Optimized Preprocessing params:',tr_params)
            model = self.__estimator(**hy_params)
            self.combos.append(self.__paramOptimizer.best_combo)

            # scale with Standard Scaler or MinMax Scaler
            scaler = tr_params['tr_scaler']()
            scaler.fit(train_predictors)
            train_predictors = scaler.transform(train_predictors)
            test_predictors = scaler.transform(test_predictors)

            # reduce dimensions w/ PCA
        #             if tr_params['PCA'] == 'PCA':
        #                     pca = PCA(n_components=tr_params['pca_n_components'],
        #                               whiten=tr_params['pca_whiten'])
        #                     pca.fit(train_predictors)
        #                     train_predictors = pca.transform(train_predictors)
        #                     test_predictors = pca.transform(test_predictors)
        #             elif tr_params['PCA'] == 'KernelPCA':
        #                     pca = KernelPCA(n_components=tr_params['pca_n_components'],
        #                                     kernel='poly')#tr_params['pca_kernel'],
        #                                     #gamma=tr_params['pca_gamma'],
        #                                     #_jobs=-1)
        #                     pca.fit(train_predictors)
        #                     train_predictors = pca.transform(train_predictors)
        #                     test_predictors = pca.transform(test_predictors)
        #             elif tr_params['PCA'] == 'SparsePCA':
        #                     pca = SparsePCA(n_components=tr_params['pca_n_components'])
        #                                     #alpha=tr_combo['pca_alpha'],
        #                                     #ridge_alpha=tr_combo['pca_ridge'],
        #                                     #_jobs=-1)
        #                     pca.fit(train_predictors)
        #                     train_predictors = pca.transform(train_predictors)
        #                     test_predictors = pca.transform(test_predictors)

            # train model
            model.fit(train_predictors,train_labels)
            # make predictions
            test_predictions = model.predict(test_predictors)
            train_predictions = model.predict(train_predictors)

            #apply more transformations
            if self.__apply:
                train_predictions=self.__applyFunc(train_predictions,train_applyData)
                test_predictions=self.__applyFunc(test_predictions,test_applyData)
                train_labels  =self.__applyFunc(train_labels,train_applyData)
                test_labels = self.__applyFunc(test_labels,test_applyData)

            # performance mearsure
            test_rmsef = rmsef(test_predictions,test_labels)
            train_rmsef = rmsef(train_predictions,train_labels)

            if self.__looCV or (len(self.__data)/self.__cv) < 10:
                test_scores['predictions'] += list(test_predictions)
                test_scores['labels'] += list(test_labels)
            else:
                test_scores['rmse'][0].append(test_rmsef)
                test_scores['rmse'][1].append(params)
                test_scores['r_squared'].append(r2score(test_predictions,test_labels))

            train_scores['rmse'][0].append(train_rmsef)
            train_scores['rmse'][1].append(params)
            train_scores['r_squared'].append(r2score(train_predictions,train_labels))

        if self.__looCV or (len(self.__data)/self.__cv) < 10:
            test_scores['rmse'] = rmsef(np.array(test_scores['predictions']),np.array(test_scores['labels']))
            test_scores['r_squared'] = r2score(test_scores['predictions'],test_scores['labels'])
        else:
            test_scores['mean_rmse'] = np.mean(test_scores['rmse'][0])
            test_scores['std_rmse'] = np.std(test_scores['rmse'][0])
            test_scores['mean_r_squared'] = np.mean(test_scores['r_squared'])
            test_scores['std_r_squared'] = np.std(test_scores['r_squared'])

        train_scores['mean_rmse'] = np.mean(train_scores['rmse'][0])
        train_scores['std_rmse'] = np.std(train_scores['rmse'][0])
        train_scores['mean_r_squared'] = np.mean(train_scores['r_squared'])
        train_scores['std_r_squared'] = np.std(train_scores['r_squared'])

        # Combination Stats
        comboStats = max_ocur(self.combos)

        if len(comboStats) < 5:
            stopper = len(comboStats)
        else:
            stopper = 5

        #Print results to terminal and text file

        #Combination stats results
        print('\n\n(((((((((((((((((((((((((((((((((((((( Combination Stats ))))))))))))))))))))))))))))))))))))))')
        print('\n\n(((((((((((((((((((((((((((((((((((((( Combination Stats ))))))))))))))))))))))))))))))))))))))',file=self.__file_)
        for combo_,ocur_,freq_ in comboStats[:stopper]:
            print('\nParams:')
            print('\nParams:',file=self.__file_)
            print('\tModel: %s'%(combo_['hyper_params']))
            print('\tModel: %s'%(combo_['hyper_params']),file=self.__file_)
            print('\tPreprocessing: %s'%(combo_['transform_params']))
            print('\tPreprocessing: %s'%(combo_['transform_params']),file=self.__file_)
            print('Stats:')
            print('Stats:',file=self.__file_)
            print('\tOcurrence: %d'%(ocur_))
            print('\tOcurrence: %d'%(ocur_),file=self.__file_)
            print('\tFrequency: %f'%(freq_))
            print('\tFrequency: %f'%(freq_),file=self.__file_)
            print('************************************************************************')
            print('************************************************************************',file=self.__file_)

        #Model Performance Results
        self.results = {'train_scores':train_scores,'test_scores':test_scores}
        print('\n\n(((((((((((((((((((((((((((((((((((((( Model Performance Results ))))))))))))))))))))))))))))))))))))))')
        print('\n\n(((((((((((((((((((((((((((((((((((((( Model Performance Results ))))))))))))))))))))))))))))))))))))))',file=self.__file_)
        print('Test Scores:')
        print('Test Scores:',file=self.__file_)
        if self.__looCV or (len(self.__data)/self.__cv) < 10:
            print('\tRMSE: %f'%(test_scores['rmse']))
            print('\tR Squared: %f'%(test_scores['r_squared']))
            print('\tRMSE: %f'%(test_scores['rmse']),file=self.__file_)
            print('\tR Squared: %f'%(test_scores['r_squared']),file=self.__file_)
        else:
            print('\tMean RMSE: %f'%(test_scores['mean_rmse']))
            print('\tSTD RMSE: %f'%(test_scores['std_rmse']))
            print('\tMean RMSE: %f'%(test_scores['mean_rmse']),file=self.__file_)
            print('\tSTD RMSE: %f'%(test_scores['std_rmse']),file=self.__file_)
            print('\tMean R Squared: %s'%(test_scores['mean_r_squared']))
            print('\tSTD R Squared: %f'%(test_scores['std_r_squared']))
            print('\tMean R Squared: %s'%(test_scores['mean_r_squared']),file=self.__file_)
            print('\tSTD R Squared: %f'%(test_scores['std_r_squared']),file=self.__file_)
            test_rmse, params_rmse = test_scores['rmse']
            print('\tRMSE SCORES:')
            print('\tRMSE SCORES:',file=self.__file_)
            for rmse_in in range(len(test_rmse)):
                print('\t\t[%d] %f  %s'%(rmse_in,test_rmse[rmse_in],params_rmse[rmse_in]))
                print('\t\t[%d] %f  %s'%(rmse_in,test_rmse[rmse_in],params_rmse[rmse_in]),file=self.__file_)
        print('\nTrain Scores:')
        print('\nTrain Scores:',file=self.__file_)
        print('\tMean RMSE: %f'%(train_scores['mean_rmse']))
        print('\tSTD RMSE: %f'%(train_scores['std_rmse']))
        print('\tMean RMSE: %f'%(train_scores['mean_rmse']),file=self.__file_)
        print('\tSTD RMSE: %f'%(train_scores['std_rmse']),file=self.__file_)
        print('\tMean R Squared: %s'%(train_scores['mean_r_squared']))
        print('\tSTD R Squared: %f'%(train_scores['std_r_squared']))
        print('\tMean R Squared: %s'%(train_scores['mean_r_squared']),file=self.__file_)
        print('\tSTD R Squared: %f'%(train_scores['std_r_squared']),file=self.__file_)
        train_rmse, params_rmse = train_scores['rmse']
        print('\tRMSE SCORES:')
        print('\tRMSE SCORES:',file=self.__file_)
        for rmse_in in range(len(train_rmse)):
            print('\t\t[%d] %f  %s'%(rmse_in,train_rmse[rmse_in],params_rmse[rmse_in]))
            print('\t\t[%d] %f  %s'%(rmse_in,train_rmse[rmse_in],params_rmse[rmse_in]),file=self.__file_)
        print('************************************************************************')
        print('************************************************************************',file=self.__file_)

        return test_scores['mean_rmse'], test_scores['mean_r_squared']
class AlloyModelEval():
    '''
    DESCRIPTION: Tests model performance on a given dataset using cross validation and inner grid search cross validation.
    Saves results in a text file.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        eval_name: str
            Name of test (a directory and text file will be created with this name)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        estimator : ML algorithm (must have functions fit() and predict())
            Estimator is the ML algorithm to be tested. Any ML algorithm defined with fit() and predict() will work.
            predict() must return a numpy ndarray.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        alloy_data : dict {'preds':predictors,'labels':labels}
            -predictors are contained in a pandas DataFrame
            -labels are contained in a numpy Series

            !From AlloyDataPreper's prep_it() return value!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model_param_grid : dict
            A dictionary containing parameter values to try for parameters of model.

            RandomForestRegressor EX: mpg = {'n_estimators':[100,200,300],'max_features':[10,15,20]}
            Will instantiate model with following combinations : n_estimators=100,max_features=10
                                                                 n_estimators=100,max_features=15
                                                                                 .
                                                                                 .
                                                                                 .
                                                                 n_estimators=300,max_features=20
            ! Will be used in myGSCV class only for param optimization !
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tran_param_grid : dict
            A dictionary containing preprocessing values to try.

            EX: tpg = {'tr_scaler':[StandardScaler,MinMaxScaler]}

            ! Will be used in myGSCV class only for param optimization !
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        apply : bool [default=False]
            If True, will transform predictons mabe by model and corresponding labels with applyFunc provided.

            !ApplyFunc and DataPreper must be provided!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        DataPreper : function [default=None]
            Function to combine predictors and labels into on DataFrame and prepare applyData used in applyFunc.

            Must take follwoing parameters:

                def DataPreper(alloyData):
                    ...

            Must always return in this format:

                return (pred_label_data,label_name,apply_data)

            where pred_label_data is predictors and labels combined in a DataFrame, label_name is name of label,
            and apply_data is the data that will be passed to applyFunc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        applyFunc : function [default=None]
            Function to be applied to predictions made by model and corresponding labels.
            Useful if applying log to labels as an example, need to unlog the predictions and labels before
            calculating performance measures (RMSE and R-Squared).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cv : int [default=10]
            Number of outer folds to perform (KFold).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        gscv : int [default=10]
            Number of inner folds to perform (KFold).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        looCV : bool [default=False]
            If True, will perform Leave one out on outer cross validation.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_jobs : int [default=-1]
            Number of cores to use. if n_jobs==-1, will use all available cores.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fileMode : str [deafult='w']
            File mode to open file with.
    '''

    def __init__(self,eval_name,
                      estimator,
                      alloy_data,
                      model_param_grid,
                      tran_param_grid,
                      apply=False,
                      DataPreper=None,
                      applyFunc=None,
                      cv=10,
                      gscv=10,
                      looCV=False,
                      n_jobs=-1,
                      fileMode='w'):
        '''[summary]

        Arguments:
            eval_name {[type]} -- [description]
            estimator {[type]} -- [description]
            alloy_data {[type]} -- [description]
            model_param_grid {[type]} -- [description]
            tran_param_grid {[type]} -- [description]

        Keyword Arguments:
            apply {bool} -- [description] (default: {False})
            DataPreper {[type]} -- [description] (default: {None})
            applyFunc {[type]} -- [description] (default: {None})
            cv {int} -- [description] (default: {10})
            gscv {int} -- [description] (default: {10})
            looCV {bool} -- [description] (default: {False})
            n_jobs {int} -- [description] (default: {-1})
            fileMode {str} -- [description] (default: {'w'})
        '''

        # Model
        self.__estimator = estimator
        # Data
        self.__alloy_data = alloy_data
        self.__combined_data = None
        #!self.label_name = alloy_data['labels'].name
        self.__data_name = alloy_data['name']
        # Evaluation Tag
        self.__eval_name = eval_name
        # File settings
        self.__file = eval_name+'.txt'
        self.__fileMode = fileMode
        # Directory
        self.__DirPath = os.path.join(RESULTS_PATH,eval_name)
        # Params to try
        self.__model_param_grid = model_param_grid
        self.__tran_param_grid = tran_param_grid
        # Cpu core usage
        self.__n_jobs=n_jobs
        # Cross Validation
        self.__looCV = looCV
        self.__gscv = gscv
        self.__cv = cv
        self.__cvTool = None
        # Apply Transformers
        self.__apply = apply
        self.__applyFunc = applyFunc
        self.__DataPreper = DataPreper

    def perform_validation(self):
        '''
        DESCRIPTION: Tests model performance on a given dataset using cross validation and inner grid search cross validation.
        Saves results in a text file.
        '''
        #Make directory if one does not exist for this test
        if os.path.isdir(self.__DirPath) == False:
            os.mkdir(self.__DirPath)
            os.chdir(self.__DirPath)
        elif os.getcwd() != self.__DirPath:
            os.chdir(self.__DirPath)



        fileE = open(self.__file,self.__fileMode)
        print('Evaluation Test on dataset:',self.__data_name,file=fileE)
        print('Evaluation Test on dataset:',self.__data_name)

        print('\nTest:',self.__eval_name)
        print('\nTest:',self.__eval_name,file=fileE)

        if self.__apply:
            pred_label_data,label_name,apply_data = self.__DataPreper(self.__alloy_data)
            print('Size of Dataset: ',len(pred_label_data))
            print('Size of Dataset: ',len(pred_label_data),file=fileE)
            #corrMatrix = pred_label_data.corr()

            self.__cvTool = CrossVal(estimator=self.__estimator,
                                    data=pred_label_data,
                                    label=label_name,
                                    model_param_grid=self.__model_param_grid,
                                    tran_param_grid=self.__tran_param_grid,
                                    file_=fileE,
                                    apply=self.__apply,
                                    applyFunc=self.__applyFunc,
                                    applyData=apply_data,
                                    cv=self.__cv,
                                    gscv=self.__gscv,
                                    looCV=self.__looCV)
        else:
            label_name = self.__alloy_data['labels'].name
            pred_label_data = self.__alloy_data['preds']
            pred_label_data[label_name] = self.__alloy_data['labels']
            print('Size of Dataset: ',len(pred_label_data))
            print('Size of Dataset: ',len(pred_label_data),file=fileE)
            #corrMatrix = pred_label_data.corr()

            self.__cvTool = CrossVal(estimator=self.__estimator,
                                    data=pred_label_data,
                                    label=label_name,
                                    model_param_grid=self.__model_param_grid,
                                    tran_param_grid=self.__tran_param_grid,
                                    file_=fileE,
                                    cv=self.__cv,
                                    gscv=self.__gscv,
                                    looCV=self.__looCV)


        # perform evaluation
        mean_rmse,mean_r2= self.__cvTool.validate()
        fileE.close()
        return mean_rmse, mean_r2

#Other utility functions
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300,location=RESULTS_PATH):
    '''
    DESCRIPTION: Saves figure created using matplotlib.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        fig_id : str
            Name of figure
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tight_layout : bool
            ***
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig_extension : str [default='png']
            ***
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        resolution : int
            Resolution of figure.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        location : str [default=RESULTS_PATH]
            Location to save figure in.

    '''
    path = os.path.join(location, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def max_ocur(combos):
    '''
    DESCRIPTION: Determines most frequent combinations used in CrossVal class.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        combos : list of dict
            A list containing all optimized combinations used during model performance testing.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RETURNS
        list
            list sorted from most frequent to least,
            with each element containing a tuple of the combination, number of occurences, and frequency
    '''
    totalcombos = len(combos)
    combo = []
    comboOcur = []
    for com in combos:
        if com not in combo:
            combo.append(com)
            comboOcur.append(1)
        else:
            comboOcur[combo.index(com)] += 1
    comSorted = list(zip(combo,comboOcur))
    comSorted.sort(key=lambda x:x[-1],reverse=True)
    for i in range(len(comSorted)):
        comSorted[i] += tuple([comSorted[i][1]/totalcombos])
    return comSorted

def rmsef(predictions,targets):
    '''
    Returns float : computes rmse score
    '''
    differences = (predictions - targets)# / predictions       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val

def r2score(x,y):
    '''
    Returns float : computes r-squared score
    '''
    slope, intercept, r_val, p_val, std_err = stats.linregress(x,y)
    return r_val**2

def make_hist(readyData,graph_name):
    combined = readyData['preds']
    combined[readyData['labels'].name] = readyData['labels']
    combined.hist(bins=100,figsize=(20,15))
    save_fig(graph_name)

def prep_poly_data(readyData,deg):
    '''
    DESCRIPTION: Prepares data for multivarite polynomial regression algorithm.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PARAMETERS:

        readyData : DataFrame
            Data to expand.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        deg : int
            degree to apply to data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RETURNS:
        DataFrame
            Data has been expanded to degree deg.

    '''
    pf = PolynomialFeatures(degree=deg)
    pred_poly = pf.fit_transform(readyData['preds'])
    df_ver = pd.DataFrame(pred_poly,index=readyData['preds'].index)
    readyData['preds'] = df_ver
    return readyData

#!################################ DATA PREPARATION #####################################################

#! Data Preper
def Data_9_12_Preper(alloyData):
    label_name = 'Temp*log(RT)'
    predictors = alloyData['preds'].copy()
    labels = alloyData['labels'].copy()

    pred_label_data = predictors.drop(['CT Temp'],axis=1)
    pred_label_data[label_name] = pd.Series(predictors['CT Temp'] * np.log(labels),index=predictors.index)
    apply_data = predictors['CT Temp']

    return (pred_label_data,label_name,apply_data) #Must always return in this format

#! Apply Function
def apply_9_12_func(data,apply_data):
    tr_data = data / apply_data
    final_data = np.exp(tr_data)
    return final_data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#! Data Preper
def Data_9_12_Preper_log(alloyData):
    predictors = alloyData['preds'].copy()
    labels = alloyData['labels'].copy()

    pred_label_data = predictors
    pred_label_data[labels.name] = labels
    apply_data = pd.Series([1]*len(labels))

    label_name = labels.name

    return (pred_label_data,label_name,apply_data) #Must always return in this format

#! Apply Function
def apply_9_12_func_log(data,apply_data):
    return np.exp(data)

class Backward_Selection():
    def __init__(self,
                fill_vals,
                dropna,
                exclude,
                t_pg,
                m_pg,
                estimator,
                eval_name,
                dataset_name='9_12_Cr.csv',
                label='RT'
                ):

        self.fill_vals = fill_vals
        self.exclude = exclude
        self.dropna = dropna
        self.dataset_name = dataset_name
        self.label = label
        self.data = None

        self.t_pg = t_pg
        self.m_pg = m_pg
        self.estimator = estimator
        self.eval_name = eval_name
        self.evaluator = None

    def prepare_data(self):
        datapreper = AlloyDataPreper(Dataset=self.dataset_name,
                            label=self.label,
                            dropna_cols=self.dropna,
                            exclude_cols=self.exclude,
                            fill_vals=self.fill_vals,
                            )
        self.data = datapreper.prep_it_split()

    def get_features(self):
        return self.data['preds'].columns.tolist()

    def prepare_evaluator(self):
        self.evaluator = AlloyModelEval(eval_name=self.eval_name,
                                estimator=self.estimator,
                                alloy_data=self.data,
                                model_param_grid=self.m_pg,
                                tran_param_grid=self.t_pg,
                                cv=10,
                                gscv=5
                                )

    def select(self):
        #initial try
        self.prepare_data()
        try_features = self.get_features()
        self.prepare_evaluator()
        mean_rmse,mean_r2 = self.evaluator.perform_validation()

        next_min_rmse = mean_rmse
        next_min_r2 = mean_r2

        cur_min_rmse = mean_rmse
        cur_min_r2 = mean_r2

        final_drop_feat = ''


        while True:
            for f in try_features:
                self.exclude.append(f)
                self.prepare_data()
                self.prepare_evaluator()

                reduced_mean_rmse,reduced_mean_r2 = self.evaluator.perform_validation()
                if reduced_mean_rmse < next_min_rmse:
                    final_drop_feat = f
                    next_min_rmse = reduced_mean_rmse
                    next_min_r2 = reduced_mean_r2
                self.exclude.remove(f)

            if next_min_rmse < cur_min_rmse:
                cur_min_rmse = next_min_rmse
                cur_min_r2 = next_min_r2
                self.exclude.append(final_drop_feat)
                try_features.remove(final_drop_feat)
            else:
                break

        return cur_min_rmse, cur_min_r2, self.exclude



if __name__ == "__main__":

    #%%
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

    dropna9_12Cr_3 = ['CT Temp','CS','RT','AGS No.','EL','RA_2','1.0% CS','2.0% CS','5.0% CS']


    #Features/Columns to remove from dataset
    exclude9_12Cr = ['MCR','0.5% CS','1.0% CS','2.0% CS','5.0% CS',
                    'UTS','Elong',
                    'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                    'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']#'B','Co','Temper2','Temper1']

    exclude9_12Cr_4 = ['MCR','0.5% CS','1.0% CS','2.0% CS','5.0% CS',
                    'Normal','Fe','Cr','N','AGS','V','Mn','C','B','P','Si','Ni','Nb','S','Mo', #dropping 1.0-5.0% CS features
                    'UTS','Elong',
                    'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                    'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']

    exclude9_12Cr_2 = ['MCR','0.5% CS','1.0% CS','2.0% CS','5.0% CS',
                  'UTS','Elong',
                  'Normal','Fe','Cr','N','AGS No.','V','Mn','C','B','RA_2','Temper1','P','Si','Ni','Nb','1.0% CS','S','Mo', #recommended to remove based on vif
                  'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                  'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']#'B','Co','Temper2','Temper1']
    exclude9_12Cr_3 = ['MCR','0.5% CS',
                  'UTS','Elong',
                  'Normal','Fe','Cr','N','AGS','V','Mn','C','B','P','Si','Ni','Nb','S','Mo', #recommended to remove based on CorMat
                  'TT Temp','YS','RA','0.1% CS','0.2% CS','TTC',
                  'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']#'B','Co','Temper2','Temper1']

    N9_12Cr = AlloyDataPreper(Dataset='9_12_Cr.csv',#name of dataset must match name of csv file located in RESULTS_PATH
                            label='RT',
                            dropna_cols=dropna9_12Cr,
                            exclude_cols=exclude9_12Cr,
                            fill_vals=fillvals,
                            )

    N9_12Cr_Reduced = AlloyDataPreper(Dataset='9_12_Cr.csv',#name of dataset must match name of csv file located in RESULTS_PATH
                            label='RT',
                            dropna_cols=dropna9_12Cr,
                            exclude_cols=exclude9_12Cr_4,
                            fill_vals=fillvals,
                            )
    ready9_12Cr = N9_12Cr.prep_it()
    ready9_12Cr_Reduced = N9_12Cr_Reduced.prep_it()


    #!################################ END #####################################################




    # #%%
    # ###### Corr Matrix Calculations
    # corMat = ready9_12Cr['data'].corr()
    # os.chdir(RESULTS_PATH)
    # f = open('corrMat9-12Cr.txt','w')
    # #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(corMat.to_string(),file=f)



    #%%
    # Initial Testing #

    # Random Forest Regression
    # print(ready9_12Cr['data'].head(50))

    # #! Transformer Param Grid
    # t_pg = [
    #         {'tr_scaler':[StandardScaler,MinMaxScaler]}]
    # #! Model Param Grid
    # m_pg = {'n_estimators':[325],'n_jobs':[-1]}
    # #? 9_12Cr
    # Evaluator = AlloyModelEval(eval_name='RFR 9-12Cr test',#Give this test a name
    #                         estimator=RandomForestRegressor,#Give it an ML algorithm to use
    #                         alloy_data=N9_12Cr.prep_it_split(),#Give it the Data preped
    #                         model_param_grid=m_pg,#Give it parameter values to try
    #                         tran_param_grid=t_pg,#Give it scalers to try
    #                         cv=10,
    #                         gscv=5
    #                         )
    # Evaluator.perform_validation()

    # Evaluator = AlloyModelEval(eval_name='RFR 9-12Cr test reduced',#Give this test a name
    #                         estimator=RandomForestRegressor,#Give it an ML algorithm to use
    #                         alloy_data=N9_12Cr_Reduced.prep_it_split(),#Give it the Data preped
    #                         model_param_grid=m_pg,#Give it parameter values to try
    #                         tran_param_grid=t_pg,#Give it scalers to try
    #                         cv=10,
    #                         gscv=5
    #                         )
    # Evaluator.perform_validation()


    # # Linear Regession
    # #! Transformer Param Grid
    # t_pg = [{'tr_scaler':[StandardScaler,MinMaxScaler]}]
    # #! Model Param Grid
    # m_pg = {}

    # #? CREEP
    # Evaluator = AlloyModelEval(eval_name='LinearReg 9-12Cr test',
    #                             estimator=LinearRegression,
    #                             alloy_data=N9_12Cr.prep_it_split(),
    #                             model_param_grid=m_pg,
    #                             tran_param_grid=t_pg,
    #                             cv=10,
    #                             gscv=5
    #                             )
    # Evaluator.perform_validation()

    # Evaluator = AlloyModelEval(eval_name='LinearReg 9-12Cr test reduced',
    #                             estimator=LinearRegression,
    #                             alloy_data=N9_12Cr_Reduced.prep_it_split(),
    #                             model_param_grid=m_pg,
    #                             tran_param_grid=t_pg,
    #                             cv=10,
    #                             gscv=5
    #                             )
    # Evaluator.perform_validation()


    # Multi-Layer Perceptron Neural Network Regression
    t_pg = [{'tr_scaler':[StandardScaler]}]
    #! Model Param Grid
    # m_pg = {'max_iter':[250,300,350,400],
    #         'activation':['relu'],
    #         'solver':['lbfgs'],
    #         'alpha':[0.0001,0.001,0.01,0.1,1,10],
    #         'learning_rate':['constant','invscaling','adaptive']}











    m_pg = {'max_iter':[300],
                'activation':['relu'],
                'solver':['lbfgs'],
                'alpha':[0.01],
                'learning_rate':['adaptive']}


    # Evaluator = AlloyModelEval(eval_name='MLPReg 9-12Cr test 4 cormat',
    #                             estimator=MLPRegressor,
    #                             alloy_data=N9_12Cr.prep_it_split(),
    #                             model_param_grid=m_pg,
    #                             tran_param_grid=t_pg,
    #                             cv=10,
    #                             gscv=5
    #                             )
    # Evaluator.perform_validation()

    # Evaluator = AlloyModelEval(eval_name='MLPReg 9-12Cr test reduced 4 cormat',
    #                             estimator=MLPRegressor,
    #                             alloy_data=N9_12Cr_Reduced.prep_it_split(),
    #                             model_param_grid=m_pg,
    #                             tran_param_grid=t_pg,
    #                             cv=10,
    #                             gscv=5
    #                             )
    # Evaluator.perform_validation()
    bs = Backward_Selection(fillvals,dropna9_12Cr,exclude9_12Cr,t_pg,m_pg,MLPRegressor,'back_select test MLPReg')
    print(bs.select())
    #TODO add forward selection capabilities