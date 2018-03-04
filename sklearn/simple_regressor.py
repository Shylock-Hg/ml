#! /home/shylock/App/miniconda3/envs/mltoolchain/bin/python

import sys
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

PATH_OECE_BLI = './dataset/oecd_bli_2015.csv'
PATH_GDP_PER_CAPITA='./dataset/gdp_per_capita.csv'

def prepare_country_stats(oecd_bli,gdp_per_capita):
    '''
        @brief combine two dataframe then slice by 
            **feature**['GDP per capita','Life satisfaction'] & 
            **indices**[keep_indices]
        @param oecd_bli
        @param gdp_per_capita
        @retval dataframe['GDP per capita','Life satisfaction']
    '''
    #select feature INEQUALITY==TOT
    oecd_bli = oecd_bli[oecd_bli['INEQUALITY']=='TOT']
    #reformat dataframe to [Country,Indicator(include 'Life satisfaction')]
    oecd_bli = oecd_bli.pivot(index='Country',columns='Indicator',values='Value')
    #print(oecd_bli)

    #rename column name : '2015'-->'GDP per capita'
    gdp_per_capita.rename(columns={'2015':'GDP per capita'},inplace=True)
    #reformat dataframe index to [Country,columns]
    gdp_per_capita.set_index('Country',inplace=True)
    #print(gdp_per_capita)
    
    #merge dataframe to [Country,columns]
    full_country_stats = pd.merge(left=oecd_bli,right=gdp_per_capita,
            left_index=True,right_index=True)
    full_country_stats.sort_values(by='GDP per capita',inplace=True)
    #print(full_country_stats)

    #remove example by indices 
    remove_indices = set([0,1,6,8,33,34,35])
    keep_indices = list(set(range(36))-remove_indices)

    #print(full_country_stats['Life satisfaction'])

    #slice by **feature**['GDP per capita','Life satisfaction'] & **indices**[keep_indices]
    return full_country_stats[['GDP per capita','Life satisfaction']].iloc[keep_indices]

'''
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli['INEQUALITY']=='TOT']
    oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')
    gdp_per_capita.rename(columns={'2015': 'GDP per capita'}, inplace=True)
    gdp_per_capita.set_index('Country', inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by='GDP per capita', inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[keep_indices]
'''

def main():
    #handle arguments from terminal
    argparser = argparse.ArgumentParser()
    argparser.add_argument('method',help='Choose a regressor from <linear> or <kNN>.')
    args = argparser.parse_args()

    #check args
    '''
    valid_params = ['linear','kNN']
    if 2!=argc :
        raise Exception('Invalid count of parameter:{}'.format(argc))
    elif argv[1] not in valid_params :
        raise Exception('Invalid parameter value:{}'.format(argv[1]))
    '''

    #load the data
    oecd_bli = pd.read_csv(PATH_OECE_BLI,thousands=',');
    gdp_per_capita = pd.read_csv(PATH_GDP_PER_CAPITA,thousands=',',delimiter='\t',
            encoding='latin1',na_values='n/a');

    #prepare the data
    country_stats = prepare_country_stats(oecd_bli,gdp_per_capita)
    X = np.c_[country_stats['GDP per capita']]
    y = np.c_[country_stats['Life satisfaction']]

    #visualize the data
    country_stats.plot(kind='scatter',x='GDP per capita',y='Life satisfaction')
    plt.show()

    if 'linear' == args.method:
        #select a linear regressor
        regressor = sklearn.linear_model.LinearRegression()
    elif 'kNN' == args.method:
        #select a kNN regressor
        regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
    else:
        raise Exception('Inavalid parameter : {}'.format(args.method))

    #train a regressor by [X,y]
    regressor.fit(X,y)

    #make a prediction for Cyprus
    Cyprus = [[22587]] #Cyprus's GDP per capita
    print(regressor.predict(Cyprus))  ##output the prediction

if '__main__' == __name__:
    main()

