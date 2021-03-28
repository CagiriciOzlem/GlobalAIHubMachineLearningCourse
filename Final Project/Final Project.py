




"""
Course Date: 22.03.2021
Name: Özlem
Surname: Çağırıcı
Email: ozlemilgun@gmail.com


"""


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.axes  as ax

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,classification_report,accuracy_score,mean_squared_error,r2_score

!pip install catboost
!pip install LightGBM
!pip install XGBoost

from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore",category=FutureWarning  )
 

 
# Read csv

df = pd.read_csv("C:/Users/Toshiba/Desktop/diamonds.csv")
df = df.drop("index",axis=1)

# Describe our data for each feature and use .info() for get information about our dataset

df.info()
df.describe()
df["price"].value_counts()
df["price"].value_counts().plot(kind="bar")


""" Target: 4 categorical values (Needs to be transformed with Label Encoder)
    Very Low     34663
    Low          11271
    Medium        4109
    High          2308
    Very High     1589
    Name: price, dtype: int64
"""

""" carat - Y- Z..: Outlier values""" 

df.duplicated().sum()
df.drop_duplicates(inplace = True)

df.isna().sum() """No missing value"""


# Do we need to generate new features?

"""
 - Target: 4 categorical values (Needs to be transformed with Label Encoder)
 - There are some categorical values that needs to be transformed with OneHot Encoder

"""
label_encoder = LabelEncoder()
df["Price_Label"] = label_encoder.fit_transform(df["price"]) 
df.head()

df.select_dtypes("object")

"""
 cut - color- clarity 
"""

dms = pd.get_dummies(df[["cut","color","clarity"]])     
dms.columns

X_1 = df.drop(["cut","color","clarity","price","Price_Label"],axis=1)
X = pd.concat([X_1,dms[["cut_Fair","color_D","clarity_I1"]]],axis=1)
Y = df[["Price_Label"]]
Y["Price_Label"].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
corr = X.corr()

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

 
# define oversampling strategy

!pip install -U imbalanced-learn
import imblearn


from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, Y)

 
#Split dataset into train and test sets. (0.7/0.3)

X_train, X_test, Y_train, Y_test = train_test_split(X_over,y_over, test_size=0.3,random_state = 42)
Y_train["Price_Label"].value_counts()
Y_test["Price_Label"].value_counts()



# Import Decision Tree, define different hyperparamters and tune the algorithm.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier()

?clf
clf_model = clf.fit(X_train,Y_train)
 

clf_params = {"max_depth":[2,3,5,8],
              "min_samples_split":[1,5,10,15]}
 

gridsrc = GridSearchCV(clf,clf_params,cv=5,verbose=2,n_jobs=-1) 
clf_cv_model =gridsrc.fit(X_train,Y_train)

clf_cv_model.best_params_

"""
    Out[271]: {'max_depth': 8, 'min_samples_split': 5}
"""
clf_tunned = DecisionTreeClassifier(max_depth = 8,min_samples_split =  5)
clf_model_tuned = clf_tunned.fit(X_train,Y_train)
 
Y_pred = clf_model_tuned.predict(X_test)    
Y_pred[0:5] 

 
 
# Create confusion matrix and calculate accuracy, recall, precision and f1 score.

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score,confusion_matrix
 
 
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("F1 Score = {}".format(f1_score(Y_test, Y_pred,average='macro')))

"""
    Precision = 0.7430514490415826
    Recall = 0.5692838314786526
    Accuracy = 0.8896469318532223
    F1 Score = 0.5537331734304736

"""


# Visualize feature importances.
clf_model_tuned.feature_importances_ 

 
feature_imp =   pd.DataFrame({"Importance":clf_model_tuned.feature_importances_ *100},
                            index=X_train.columns)


my_colors = 'rgbkymcb'  
 
feature_imp.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color=my_colors)

plt.xlabel("Değişken Önem Skorları")
plt.xlabel("Değişkenler")
plt.yticks(size=6)
plt.title("Değişken Önem Düzeyleri")
plt.gca().legend_=None
 
 
#The Features (High Importance Degree)

 
X_over2 = X_over[["carat","y","z","clarity_I1","color_D"]]
 
 
#Split dataset into train and test sets. (0.7/0.3)

X_train, X_test, Y_train, Y_test = train_test_split(X_over2,y_over, test_size=0.3,random_state = 42)
Y_train["Price_Label"].value_counts()
Y_test["Price_Label"].value_counts()


 
clf = DecisionTreeClassifier()

clf_params = {"max_depth":[2,3,5,8,12,20],
              "min_samples_split":[1,5,10,15,100,200]}
 

gridsrc = GridSearchCV(clf,clf_params,cv=5,verbose=2,n_jobs=-1) 
clf_cv_model =gridsrc.fit(X_train,Y_train)

clf_cv_model.best_params_

 
clf_tunned = DecisionTreeClassifier(max_depth = 20,min_samples_split =  15)
clf_model_tuned = clf_tunned.fit(X_train,Y_train)
 
Y_pred = clf_model_tuned.predict(X_test)    

# Create confusion matrix and calculate accuracy, recall, precision and f1 score.
 
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("F1 Score = {}".format(f1_score(Y_test, Y_pred,average='macro')))

"""
    Precision = 0.7046936825589105
    Recall = 0.6470620738644746
    Accuracy = 0.9022510021584952
    F1 Score = 0.6621284437520782

"""


# Visualize feature importances.
clf_model_tuned.feature_importances_ 

 
feature_imp =   pd.DataFrame({"Importance":clf_model_tuned.feature_importances_ *100},
                            index=X_train.columns)

my_colors = 'rgbkymcb'  
feature_imp.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color=my_colors)

plt.xlabel("Değişken Önem Skorları")
plt.xlabel("Değişkenler")
plt.yticks(size=6)
plt.title("Değişken Önem Düzeyleri")
plt.gca().legend_=None

 # %%
 
 
 
from xgboost import XGBClassifier

 
xgb =  XGBClassifier(random_state = 42)
xgb_params = {"n_estimators":[100,500,2000],
              "subsample":[0.6,0.8,1],
              "max_depth": [3,5],
              "learning_rate": [0.001,0.1,0.01]
                 }
?xgb
gridsrc = GridSearchCV(xgb,xgb_params,cv=5,verbose=2,n_jobs=-1) 
xgb_cv_model =gridsrc.fit(X_train,Y_train)

xgb_cv_model.best_params_ 

 
xgb =  XGBClassifier(learning_rate = 0.001,subsample = 0.8,max_depth = 5,n_estimators = 2000,random_state = 42,verbose=2,n_jobs=-1)
xgb_model_tuned= xgb.fit(X_train,Y_train)   
 
Y_pred = xgb_model_tuned.predict(X_test)    


# Create confusion matrix and calculate accuracy, recall, precision and f1 score.
 
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("F1 Score = {}".format(f1_score(Y_test, Y_pred,average='macro')))


"""
    Precision = 0.7102091853218389
    Recall = 0.5861181094411774
    Accuracy = 0.8976641998149861
    F1 Score = 0.5794759987435443

"""
 # %%
 
 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rf =  RandomForestClassifier(random_state = 42)
rf_params = {"n_estimators":[25,50,100,160],
              "min_samples_split": [2,4,5,20],
              "max_features": [3,5,6,7,8]
              }
 
gridsrc = GridSearchCV(rf,rf_params,cv=5,verbose=2,n_jobs=-1) 
rf_cv_model =gridsrc.fit(X_train,Y_train)

rf_cv_model.best_params_ 
    
rf = RandomForestClassifier(n_estimators=160,max_features=6,min_samples_split=6,verbose=2,n_jobs=-1)  
rf_model_tuned = rf.fit(X_train,Y_train)

  
Y_pred = rf_model_tuned.predict(X_test)    


# Create confusion matrix and calculate accuracy, recall, precision and f1 score.
 
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='weighted')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='weighted')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("F1 Score = {}".format(f1_score(Y_test, Y_pred,average='weighted')))


#Best model for this problem is Random Forest.
 