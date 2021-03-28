# -*- coding: utf-8 -*-
"""
Course Date: 22.03.2021
Name: Özlem
Surname: Çağırıcı
Email: ozlemilgun@gmail.com

"""
**************************HOMEWORK - 2 *************************

# Import boston dataset and convert it into pandas dataframe


from sklearn.datasets import load_boston
import pandas as pd

X , Y = load_boston(return_X_y=True)
 

print(X.head())
print(Y.head())

df = pd.DataFrame(X,columns = load_boston().feature_names)
print(df.head())


# Check duplicate values and missing data

df.info()
df.duplicated().sum()
df.isna().sum()

"""No dublicated value/no missing value"""


# Visualize data for each feature (pairplot,distplot)
df.describe()
df.columns


import seaborn as sns
sns.pairplot(df)


# Draw correlation matrix

import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()

plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot = True
)
sns.set(font_scale=0.5)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',size=8
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    size=8
)

ax.set_ylim(len(corr)+0.5, -0.5) 


# Drop correlated features (check correlation matrix)
 
"""There is a high negative correlation between the DIS & Nox 
    There is a high negative correlation between the DIS & Age 
"""

df.drop("DIS",axis=1,inplace=True)
df.drop("INDUS",axis=1,inplace=True)
df.columns
 

# Handle outliers (you can use IsolationForest)


# Outlier detection with Z-Score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df))
z


outliers = list(set(np.where(z > 3)[0]))
new_df = df.drop(outliers,axis = 0).reset_index(drop = False)
display(new_df)

 
Y_new = Y[list(new_df["index"])]
len(Y_new)

# Normalize data

X_new = new_df.drop('index', axis = 1)
X_new.shape
Y_new.shape

from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_scaled = StandardScaler().fit_transform(X_new)



# Split dataset into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y_new, test_size=0.3, random_state=42)

X_train.shape
Y_train.shape

X_test.shape
Y_test.shape

 
# Import ridge and lasso models from sklearn

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.metrics import mean_squared_error

# Define 5 different alpha values for lasso and fit them. Print their R^2 sore on both
# train and test.


ridge_obj = Ridge()
alpha_Val =  np.random.randint(1,200,5)


for i in alpha_Val:
    ridge_obj.set_params(alpha=i) 
    Ridge_Model = ridge_obj.fit(X_train,Y_train)   
    Y_pred = Ridge_Model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
 
    print("for alpha = "+ str(i) + " : "+ str(format(RMSE,".2f"))) 
    
"""
    for alpha = 159 : 4.52
    for alpha = 74 : 4.37
    for alpha = 167 : 4.54
    for alpha = 64 : 4.35
    for alpha = 42 : 4.32
"""            

 
ls_obj = Lasso()
alpha_Val =  np.random.randint(0,20,5)


for i in alpha_Val:
    ls_obj.set_params(alpha=i) 
    Lasso_Model = ls_obj.fit(X_train,Y_train)   
    Y_pred = Lasso_Model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
 
    print("for alpha = "+ str(i) + " : "+ str(format(RMSE,".2f"))) 

""" 
 
        for alpha = 6 : 7.07
        for alpha = 5 : 6.41
        for alpha = 13 : 7.38
        for alpha = 7 : 7.38
        for alpha = 7 : 7.38

"""

# Make comment about results. Print best models coefficient.

    
""" 
   - Best alpha value is :42
            for alpha = 159 : 4.52
            for alpha = 74 : 4.37
            for alpha = 167 : 4.54
            for alpha = 64 : 4.35
            for alpha = 42 : 4.32 
"""


#RIDGE MODEL :
ridge_obj_tunned = Ridge(alpha=42)
Ridge_Model = ridge_obj.fit(X_train,Y_train)   

Y_pred_train=Ridge_Model.predict(X_train)
RMSE_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
    
Y_pred_test=Ridge_Model.predict(X_test)
RMSE_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
      

print(RMSE_train) 
print(RMSE_test)

"""
4.3221870898383505
4.321378925425808

"""

 
#LASSO MODEL :
    
    """ 
   - Best alpha value is :5
        for alpha = 6 : 7.07
        for alpha = 5 : 6.41
        for alpha = 13 : 7.38
        for alpha = 7 : 7.38
        for alpha = 7 : 7.38

    """


ls_obj_tunned = Lasso(alpha=5)
Lasso_Model = ridge_obj.fit(X_train,Y_train)   

Y_pred_train=Lasso_Model.predict(X_train)
RMSE_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
    
Y_pred_test=Lasso_Model.predict(X_test)
RMSE_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
      
print(RMSE_train) 
print(RMSE_test)


