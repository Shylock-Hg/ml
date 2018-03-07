#! /home/shylock/App/miniconda3/envs/mltoolchain/bin/python

import os
import tarfile
from six.moves import urllib
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

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
    #print(housing_data.head()) 
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
        print(housing_dist-test_dist)
    elif 'stratified' == args.split:
        #category the value of 'median_income'
        housing_data['income_cat'] = np.ceil(housing_data['median_income']/1.5)
        #limit to 5
        housing_data['income_cat'].where(housing_data['income_cat']<5, 5.0, inplace=True)
        #split random by value of 'income_cat', stratified to 
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)       
        #stratified simpling by housing_data['income_cat']
        for train_index,test_index in split.split(housing_data, housing_data['income_cat']):
            strat_train_set = housing_data.loc[train_index]
            strat_test_set = housing_data.loc[test_index]

        #overview split
        strat_dist = (strat_test_set['income_cat'].value_counts()/len(strat_test_set))
        housing_dist = (housing_data['income_cat'].value_counts()/len(housing_data))
        print(housing_dist-strat_dist)


if '__main__' == __name__:
    main()

