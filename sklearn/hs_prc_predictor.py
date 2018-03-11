#! /home/shylock/App/miniconda3/envs/mltoolchain/bin/python

import os
import tarfile
from six.moves import urllib
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer,StandardScaler,LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

URL_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_FILE = 'housing.tgz'
HOUSING_URL = URL_ROOT + HOUSING_PATH + '/' + HOUSING_FILE

def fetch_housing_data(housing_url = HOUSING_URL, 
        housing_path = HOUSING_PATH,):
    '''
        @brief fetch housing.tgz & extract to HOUSING_PATH
        @param housing_url the url of data
        @param housing_path the path to save data
    '''
    #check dir/file
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,'housing.tgz')

    #download
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,'housing.csv')
    return pd.read_csv(csv_path)

class DataFrameSelector(BaseEstimator,TransformerMixin):
    '''
        @brief select attributes from dataframe
    '''
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        '''
            @brief select features frome X by attributes names
            @param X the all features
        '''
        return X[self.attribute_names].values

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    '''
        @brief combine and create attributes 
    '''
    def __init__(self,rooms_id,bedrooms_id,population_id,household_id,
            is_add_bedrooms_per_room=True):
        self.rooms_id = rooms_id
        self.bedrooms_id = bedrooms_id
        self.population_id = population_id
        self.household_id = household_id
        self.is_add_bedrooms_per_room = is_add_bedrooms_per_room

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        rooms_per_household = X[:,self.rooms_id] / X[:, self.household_id]
        population_per_household = X[:,self.population_id] / X[:, self.household_id]
        if self.is_add_bedrooms_per_room:
            bedrooms_per_room = X[:,self.bedrooms_id] / X[:, self.rooms_id]
            return np.c_[X, rooms_per_household, population_per_household, 
                    bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def main():
    # handle parameter
    argparser = argparse.ArgumentParser()       
    argparser.add_argument('-f','--force',action='store_true',
            help='Force redownload housing data!')
    argparser.add_argument('-v','--visual',action='store_true',
            help='Visualize the data!')
    argparser.add_argument('-s','--split',choices=['random','stratified'],
            required=True,
            help='Random or stratified sampling!')

    args = argparser.parse_args()

    # download & extract data
    if False == args.force :  #not force redownload data
        #check file exist
        if not os.path.isdir(HOUSING_PATH):
            fetch_housing_data()
    else:
        #force redownload data
        fetch_housing_data()

    #load data
    housing_data = load_housing_data()
    #print(housing_data.shape) 
    #print(housing_data.info()) #overview of dataframe
    #print(housing_data['ocean_proximity'].value_counts()) #count instance by diff value
    #print(housing_data.describe()) #summary of numerical attributes

    #visualize the data
    if True == args.visual :
        housing_data.hist(bins=50,figsize=(20,15))
        plt.show()
    
    #split to trian/test set
    #random split
    if 'random' == args.split :
        #random split
        train_set, test_set = train_test_split(housing_data,test_size=0.2,random_state=42)

        #compare
        #category the value of 'median_income'
        housing_data['income_cat'] = np.ceil(housing_data['median_income']/1.5)
        #limit to 5
        housing_data['income_cat'].where(housing_data['income_cat']<5, 5.0, inplace=True)
        test_set['income_cat'] = np.ceil(test_set['median_income']/1.5)
        test_set['income_cat'].where(test_set['income_cat']<5, 5.0, inplace=True)
        test_dist = (test_set['income_cat'].value_counts()/len(test_set))
        housing_dist = (housing_data['income_cat'].value_counts()/len(housing_data))
        #print(housing_dist-test_dist)

    elif 'stratified' == args.split:
        #category the value of 'median_income'
        housing_data['income_cat'] = np.ceil(housing_data['median_income']/1.5)
        #limit to 5
        housing_data['income_cat'].where(housing_data['income_cat']<5, 5.0, inplace=True)
        #split random by value of 'income_cat', stratified to 
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)       
        #stratified simpling by housing_data['income_cat']
        for train_index,test_index in split.split(housing_data, housing_data['income_cat']):
            train_set = housing_data.loc[train_index]
            test_set = housing_data.loc[test_index]

        #overview split
        strat_dist = (test_set['income_cat'].value_counts()/len(test_set))
        housing_dist = (housing_data['income_cat'].value_counts()/len(housing_data))
        #print(housing_dist-strat_dist)

        #dropout 'income_cat'
        for set in (train_set,test_set):
            set.drop(['income_cat'], axis=1, inplace=True)

    #visualize the train_set
    if True == args.visual :
        train_set.plot(kind='scatter', x='longitude', y='latitude',
                alpha=0.1, 
                #radius -- population
                s=train_set['population']/100, label='population', 
                #color -- median_house_value
                c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
        plt.legend()
        plt.show()

    if True == args.visual :
        #attrs = ['median_house_value','median_income','total_rooms','housing_median_age']
        #pd.plotting.scatter_matrix(train_set[attrs], figsize=(12,8))
        train_set.plot(kind='scatter', x='median_income', y='median_house_value',
                alpha=0.1)
        plt.show()

    #split [features,label]
    train_data = dict()
    train_data['input'] = train_set.drop('median_house_value',axis=1)
    train_data['output'] = train_set['median_house_value'].copy()

    #data prepare by pipeline
    attrs_num = list(train_data['input'].drop('ocean_proximity',1))
    attrs_cat = ['ocean_proximity']
    
    #digital attributes prepare pipeline
    num_pipeline = Pipeline([
        ('selector',DataFrameSelector(attrs_num)),
        ('imputer',Imputer(strategy='median')),
        ('attribs_adder',CombinedAttributesAdder(3,4,5,6)),
        ('std_scaler',StandardScaler()),
            ])
    #nodigital attributes prepare pipelin
    nonum_pipeline = Pipeline([
        ('selector',DataFrameSelector(attrs_cat)),
        #('label_binarizer',LabelBinarizer())
        ('label_binarizer',LabelBinarizerPipelineFriendly())
            ])
    #the whole train set prepared pipeline
    prepare_pipeline = FeatureUnion(transformer_list = [
        ('num_pipeline',num_pipeline),
        ('nonum_pipeline',nonum_pipeline)
        ])

    train_data['input'] = prepare_pipeline.fit_transform(train_data['input'])
    print(train_data['input'])

    #select and train model
    #lin_reg = LinearRegression()
    #lin_reg.fit(train_data['input'],train_data['output'])

    #test
    #some_data = train_data['input'][:5]
    #some_label = train_data['output'].iloc[:5]
    #print('Predictions:\t',lin_reg.predict(some_data))
    #print('Labels:\t\t',list(some_label))

    #measure RMSE
    #predctions = lin_reg.predict(train_data['input'])
    #lin_mse = mean_squared_error(train_data['output'],predctions)
    #lin_rmse = np.sqrt(lin_mse)
    #print(lin_rmse)

    #more complex model



    '''
    #create new attibute
    train_set['rooms_per_household'] = train_set['total_rooms']/train_set['households']
    train_set['bedrooms_per_room'] = train_set['total_bedrooms']/train_set['total_rooms']
    train_set['population_per_household'] = train_set['population']/train_set['households']

    #correlations matrix of each attribute to each other
    #corr = train_set.corr()
    #print(corr['median_house_value'].sort_values(ascending=False))
    #print(train_set.head())

    #finally train data['input','output']
    train_data = dict()
    train_data['input'] = train_set.drop('median_house_value',axis=1)
    train_data['output'] = train_set['median_house_value'].copy()

    #data cleaning
    imputer = Imputer(strategy = 'median')
    train_num  = train_data['input'].drop('ocean_proximity',axis=1)
    imputer.fit(train_num)  #get median
    #print(imputer.statistics_)
    X = imputer.transform(train_num)
    train_num = pd.DataFrame(X,columns=train_num.columns)
    #print(train_num.head())

    #handle text attributes
    encoder = LabelEncoder()
    train_nonum = train_data['input']['ocean_proximity'].copy()
    train_nonum_encoded = encoder.fit_transform(train_nonum)
    #print(train_nonum_encoded.shape)
    encoder = OneHotEncoder()
    train_nonum_encoded = encoder.fit_transform(train_nonum_encoded.reshape(-1,1))
    #print(train_nonum_encoded.shape)
    #print(train_nonum_encoded.toarray())

    #feature scale
    '''

if '__main__' == __name__ :
    main()


