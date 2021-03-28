
 
"""
Course Date: 22.03.2021
Name: Özlem
Surname: Çağırıcı
Email: ozlemilgun@gmail.com
"""
**************************HOMEWORK - 3 *************************

# Import necessary libraries

from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Generate dataset using make_classification function in the sklearn. 
X, Y = make_classification(n_samples=10000  , n_features=8, n_informative=5,class_sep=2, random_state=42) 
X.shape
Y.shape


# Convert it into pandas dataframe.
X = pd.DataFrame(X)
X.columns=["A","B","C","D","E","F","G","H"]


Y = pd.DataFrame(Y, columns=["Target"])
df = pd.concat([X,Y],axis=1) 

# Check duplicate values and missing data.

df.duplicated().sum()
df.isna().sum()

# Visualize data for each feature (pairplot,distplot).
df.describe()
df.columns



sns.pairplot(df)


# Draw correlation matrix.

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


df.drop("D",axis=1,inplace=True)
df.drop("H",axis=1,inplace=True)
df.columns
 

# Handle outliers (you can use IsolationForest)


# Outlier detection with Z-Score

z = np.abs(stats.zscore(df))
 
outliers = list(set(np.where(z > 3)[0]))
df = df.drop(outliers,axis = 0).reset_index(drop = False)
 
X = df.drop(["Target"],axis=1)
Y =  new_df[["Target"]]


# Split dataset into train and test set

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)


# Import Decision Tree, define different hyperparamters and tune the algorithm.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier()
clf_model = clf.fit(X_train,Y_train)
 

clf_params = {"max_depth":[2,3,8],
              "min_samples_split":[1,5,10,20]}
 

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



# Visualize feature importances.
clf_model_tuned.feature_importances_ 

 
feature_imp =   pd.DataFrame({"Importance":clf_model_tuned.feature_importances_ *100},
                            index=X_train.columns)


my_colors = 'rgbkymcb'  
"""
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white

"""


feature_imp.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color=my_colors)

plt.xlabel("Değişken Önem Skorları")
plt.xlabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.gca().legend_=None
 
"""
#Importance of Features
1.E
2.B
3.A
4.G
5.F

H - C features do not contribute to model explanation. 
"""


# Create confusion matrix and calculate accuracy, recall, precision and f1 score.

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score,confusion_matrix
 
 
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("F1 Score = {}".format(f1_score(Y_test, Y_pred,average='macro')))

"""
    
    Precision = 0.9775163565027412
    Recall = 0.9775358956234588
    Accuracy = 0.9775469168900804
    F1 Score = 0.9775260159791366

"""


# Import XGBoostClassifier, define different hyperparamters and tune the algorithm.
!pip install XGBoost
 
from xgboost import XGBClassifier

 
xgb =  XGBClassifier(random_state = 42)
xgb_params = {"n_estimators":[100,500,2000],
              "subsample":[0.6,0.8,1],
              "max_depth": [3,5],
              "learning_rate": [0.001,0.1,0.01]
                 }

xgb_params
gridsrc = GridSearchCV(xgb,xgb_params,cv=10,verbose=2,n_jobs=-1) 
xgb_cv_model =gridsrc.fit(X_train,Y_train)

xgb_cv_model.best_params_ 

 """
  {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 2000, 'subsample': 0.8}
 """


xgb =  XGBClassifier(learning_rate = 0.01,subsample = 0.8,max_depth = 3,n_estimators = 2000,random_state = 42)
xgb_model_tuned= xgb.fit(X_train,Y_train)   
    
 
Y_pred = xgb_model_tuned.predict(X_test)    
Y_pred[0:5] 


# Visualize feature importances.
xgb_model_tuned.feature_importances_ 

 
feature_imp =   pd.DataFrame({"Importance":xgb_model_tuned.feature_importances_ *100},
                            index=X_train.columns)


my_colors = 'rgbkymcb'  
"""
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white

"""


feature_imp.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color=my_colors)

plt.xlabel("Değişken Önem Skorları")
plt.xlabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.gca().legend_=None
 
"""
#Importance of Features
1.E
2.A
3.B
4.G
5.F

H  does not contribute to model explanation. 
"""

 
# Create confusion matrix and calculate accuracy, recall, precision and f1 score.

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score,confusion_matrix
 
 
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("F1 Score = {}".format(f1_score(Y_test, Y_pred,average='macro')))
 
"""
    
    Precision = 0.989519016881671
    Recall = 0.9897151372244315
    Accuracy = 0.9896112600536193
    F1 Score = 0.9896035995839372

"""

# Evaluate your result and select best performing algorithm for our case.

When we evaluate the relevant values (Precision,Recall,Accuracy,F1 Score), 
it is seen that the performance of the XGBoost model is better then Decision Tree.
XGboost can be preferred for this data set. 
