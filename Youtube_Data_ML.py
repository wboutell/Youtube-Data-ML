# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:42:22 2020

@author: walid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 2020

@author: Walid Boutellis
"""

import pandas as pd
from sklearn.metrics import explained_variance_score,mean_absolute_error

# Importing the dataset
dataset = pd.read_csv("C:/Users/walid/USvideosClean.csv")

#Convert the dataset into a pandas dataframe
df= pd.DataFrame(dataset)

#-------------------------------------------------------------------------------
#PREDICTING VIRALITY OF A VIDEO BASED ON LIKES, DISLIKES, AND COMMENTS
#PART 1 LINEAR REGRESSION

#Identifying independent and dependent variables
x = dataset.iloc[:, 9:12].values #predictor dataset (independent variables)
y = dataset.iloc[:, 8].values #target dataset (dependent variable)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Fitting Multiple Linear Regression to the Training set and reporting the accuracy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score,mean_absolute_error

#Creating an instance of Linear regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test)

#Printing the metrics of the linear regression model
print('Variance score (Linear Regression): %.2f', explained_variance_score(y_test, y_pred))
print('MAE score (Linear Regression): %.2f', mean_absolute_error(y_test, y_pred))

#PART 2 DECISION TREE REGRESSION

from sklearn.tree import DecisionTreeRegressor

#Creating an instance of Decision Tree regression
regressor_dt = DecisionTreeRegressor(random_state = 0,max_depth=3)
regressor_dt.fit(x_train, y_train)
y_pred_dt=regressor_dt.predict(x_test)

#Printing the metrics of the model
print('Variance score (Decision Tree Regression): %.2f', explained_variance_score(y_test, y_pred_dt))
print('MAE score (Decision Tree Regression): %.2f', mean_absolute_error(y_test, y_pred_dt))

#PART 3 RANDOM FOREST REGRESSION

from sklearn.ensemble import RandomForestRegressor
#Creating an instance of random forest regressor
rf_regressor = RandomForestRegressor(n_estimators = 5, random_state = 0)
rf_regressor.fit(x, y)
y_pred_rf=rf_regressor.predict(x_test)
print('Variance score (Random Forest Regression): %.2f', explained_variance_score(y_test, y_pred_rf))
print('MAE score (Random Forest Regression): %.2f', mean_absolute_error(y_test, y_pred_rf))

#PART 3 GRADIENT BOOSTING REGRESSION
from sklearn.ensemble import GradientBoostingRegressor
reg_gb = GradientBoostingRegressor(random_state=0)
reg_gb.fit(x_train, y_train)
y_pred_gb= reg_gb.predict(x_test)

#Printing the metrics of the model
print('Variance score (Gradient Boosting Regression): %.2f', explained_variance_score(y_test, y_pred_gb))
print('MAE score (Gradient Boosting Regression): %.2f', mean_absolute_error(y_test, y_pred_gb))

#------------------------------------------------------------------------------------------------------
#IDENTIFYING WHETHER DISABLING COMMENTS EFFECTS VIEWS, LIKES, AND DISLIKES OF YOUTUBE VIDEOS
#Fetching subsets of the dataframe
#Fetch rows in which comments and ratings are enabled
com= df.loc[(df.comments_disabled == False) & (df.ratings_disabled == False)]

#Fetch rows in which comments are disabled and ratings are enabled
no_com= df.loc[(df.comments_disabled == True) & (df.ratings_disabled == False)]

#Calculate averages for views, likes, and dislikes for rows that have comments enabled
c_views= com.views.mean()
c_likes= com.likes.mean()
c_dislikes= com.dislikes.mean()

#Calculate averages for views, likes, and dislikes for rows that have comments disabled
nc_views= no_com.views.mean()
nc_likes= no_com.likes.mean()
nc_dislikes= no_com.dislikes.mean()

#Print all the means found above and round them to two decimal points
print("When comments are enabled, videos have the following averages: ")
print(f"Views: {round(c_views,2)}")
print(f"Likes: {round(c_likes,2)}")
print(f"Dislikes: {round(c_dislikes,2)}")

print("When comments are disabled, videos have the following averages: ")
print(f"Views: {round(nc_views,2)}")
print(f"Likes: {round(nc_likes,2)}")
print(f"Dislikes: {round(nc_dislikes,2)}")

#Import stats from the scipy library to calculate the t scores
from scipy import stats


#INDEPENDENT SAMPLE T-TEST ON AVERAGES FOUND ABOVE
print("Views: "),
stats.ttest_ind(com.views, no_com.views)

print("Likes: "),
stats.ttest_ind(com.likes, no_com.likes)

print("Dislikes: "),
stats.ttest_ind(com.dislikes, no_com.dislikes)

#-------------------------------------------------------------------------------------
#IDENTIFYING WHICH VIDEO TYPES GARNER THE MOST ATTENTION 

#Import additional libraries
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Make DF of quantitative variables 
df_quantitative = df[df['views'], df['likes'], df['dislikes'], df['comment_count']]
df_quantitative.agg(['mean'], 'median')

#Regression plot of category id and likes
ax = sns.regplot(x="category_id", y="likes", fit_reg = True, data=df, x_jitter=.1)

#getting descriptive stats
df['category_id'] = df['category_id'].to_frame()
df['likes'] = df['likes'].to_frame()

df['category_id'].describe()
df['likes'].describe()

###CREATING HEAT MAPS #########

#Grouping sets by category ID and likes, comments, views
yt_cats_likes = df.groupby('category_id')['likes'].agg(['mean', 'median'])
yt_cats_comments = df.groupby('category_id')['comment_count'].agg(['mean', 'median'])
yt_cats_views = df.groupby('category_id')['views'].agg(['mean', 'median'])


#Likes heatmap
heat_map = sns.heatmap(yt_cats_likes, cbar_kws={'label': 'Likes'}, cmap='Greens',)
plt.show()

#Comments heatmap
heat_map = sns.heatmap(yt_cats_comments, cbar_kws={'label': 'Number of Comments'}, cmap='Reds')
plt.show()

#Views heatmap
heat_map = sns.heatmap(yt_cats_views, cbar_kws={'label': 'Views'}, cmap='Blues')
plt.show()


#Find the top 10 categories 
df['category_id'].value_counts()

#Results
#10.0    6472
#26.0    4146
#23.0    3457
#22.0    3210
#25.0    2487
#28.0    2401
#1.0     2345
#17.0    2174
#27.0    1656



#Create new dataframes for likes in popular categories (nonprofits&activism and music)
dfcat_20 = df[ df['category_id'] == 20]['views']
dfcat_25 = df[ df['category_id'] == 25]['views']
dfcat_10 = df[ df['category_id'] == 10]['views']
dfcat_29= df[ df['category_id'] == 29]['views']

#Test to see difference 

stats.ttest_ind(dfcat_20, dfcat_25)
stats.ttest_ind(dfcat_10, dfcat_25)
stats.ttest_ind(dfcat_29, dfcat_25)



