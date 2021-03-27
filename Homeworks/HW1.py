1)	How would you define Machine Learning?
Machine Learning - If you define Machine Learning (ML), it is the computer modeling of systems that make predictions by making inferences from data with mathematical and statistical operations. It is a sub-branch of artificial intelligence. It contains many methods and algorithm structures. It creates algorithms that help us make inferences about a dataset.  Machine learning can help define the pattern in both structured and unstructured data and identify the stories the data tell. Machine learning is done in two different ways: Supervised Learning & Unsupervised Learning.
2)	What are the differences between Supervised and Unsupervised Learning? Specify example 3 algorithms for each of these.
Supervised learning: The observations tagged in the supervised learning process are addressed. Tags guide the algorithm to make inferences. It allows us to make inferences about what the output value will be when a new data set is given to the computer from outside by referencing the relationships between the training set and inputs and outputs. Classification and regression methods are used in supervised learning. For instance;
-	While marking the credits with NPL (Nonperforming-Loan) as 1 by looking at the characteristics of the customers who have previously borrowed, marking the customers who perform well, that is, the customers who pay their loans regularly, as 0 for determining the customers with a high risk of bankruptcy when a new loan request is received.
-	Image classification: It is used to predict whether the picture we have is a cat picture or not. Predicting whether the object in the visual is a cat by passing the data sets we have (not cat / cat) through a learning model, when a new data set is presented.
-	House price predicting.
Unsupervised learning: Unlabeled observations are used in the unsupervised learning process. Here the algorithm itself makes discoveries by trying to see the invisible. Methods such as clustering or size reduction are used for these purposes.
-	Customer segmentation: Segmenting into different groups by examining customer behavior, profitability and product characteristics in order to realize customer-specific strategies
-	Anomaly Detection.
-	Clustering DNA pattern

3)	What are the test and validation set, and why would you want to use them?
One of the important issues of machine learning is the generalization of the models. Generalization is the ability to observe how the model works in harmony not only with the data you have learned, but also with the new data that we will obtain in the future, which has not been seen before. Therefore, the learning model created should be very well generalizable to ensure that future data items are correctly classified. In other words, if our model performs well on the data that it has not seen in education, we can say that it is well generalized on the given data. For this reason,  Train/Test Sample Spllitting Cross Validation techniques are used to measure generalization. 
Train Set: The data set in which the model is trained.
The validation section is selected from the train data set. In the validation part, the applied model is tried to be improved. (hyperparameter tuning); The optimum coefficients / weights are tried to be found. After the final model is established with optimum hyperparameters, the model performance is tested with the test set. According to the problem (controlled regression or classification) using error metrics, the error between the actual values and the predicted values is examined and the generalizability of the model is measured, ie the confidence level is tested

4)	What are the main preprocessing steps? Explain them in detail. Why we need to prepare our data?
1.	Gather and import the data set
2.	Import the libraries for pre-processing
3.	Duplicate Values
4.	Check if the data-set is Imbalanced or Balanced.
5.	Data Exploration (EDA)
6.	Missing value imputation
7.	Outlier detection
8.	Feature Scaling
9.	Feature Extraction
10.	Feature Encoding (Lobal Encoding, One-Hot Encoding)
Preprocessing is so important that Model Performance Depends on Data.Missing values, outliers, and inconsistencies in the data all break the integrity of the data set and eliminate the generalizability of the model. So the model may be incomplete and over-fitting  or undefitting problems may occur. So we need to fix all these problems for more accurate results. Ideally, the data preprocessing step is very critical to get a better model output.  
 

5)	How you can explore and analyse countionus and discrete variables?
While a continuous variable consists of all real values within a certain range, individual variables can be counted. Usually discrete variables are defined as numbers, but continuous variables are defined as measurements. Probability density function (PDF) is used to describe continuous probability distributions,  while Probability mass function (PMF) is used to describe discrete probability distributions. 

6)	Analyse the plot given below. (What is the plot and variable type, check the distribution and make comment about how you can preproccess it.)
The plot show that petal width included numerical variables which are continuos. The plot is a Continuous Variable – Histogram. When we examine this graph, it is necessary to solve the outlier detection problem of the petal width variable under data pre-processing
