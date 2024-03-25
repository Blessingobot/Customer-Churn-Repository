#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

# Classifier Algorithms Libraries
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


# In[109]:


# Load Dataset

df = pd.read_csv("C:\\Users\\HP\\Desktop\\CAPSTONE Project.csv")


# In[48]:


# The Top Rows

df.head()


# In[49]:


# The Bottom Rows

df.tail()


# In[50]:


# Dimensionality of the data - the number of rows and columns
df.shape


# In[51]:


# Examine the columns/ features of the data
df.columns


# In[52]:


# investigate the dataset for anomalies and data types 
df.info()


# In[167]:


# Visualizing for missing values

print(df.isnull().sum())
plt.figure(figsize = (10, 3))
sns.heatmap(df.isnull(), cbar=True, cmap='Blues_r')


# In[53]:


# Numerical Statistical Analysis
df.describe()


# In[54]:


# Categorical Statistical Analysis

df.describe(include=["object", "bool"])


# ### Exploratory Data Analysis (EDA)
# 
# ### Univariate Analysis

# In[168]:


# check for outlier 
sns.boxplot(x=df["tenure"]);


# In[18]:


#  Visualization for Churn

print(df["Churn"].value_counts())
df['Churn'].hist(bins=20,);


# In[169]:


# Tenure Segmentation

def Tenure_Segment(tenure):
    if tenure <= 24:
        return "New"
    elif tenure <= 48:
        return "Intermediate"
    else:
        return "Long Term"
    
df["Tenure_Segment"] = df["tenure"].apply(Tenure_Segment)

print(df["Tenure_Segment"].value_counts())
plt.figure(figsize = (10, 5))
sns.countplot(x='Tenure_Segment', data=df)
plt.xlabel('tenure segment')
plt.ylabel('count of Tenure segment')
plt.title( 'Tenure Segment');


# In[21]:


df.head()


# In[195]:


def Gender_Classification(gender):
    if gender <= 0:
        return "Female"
    else:
        return "Male"
    
df["Gender_Classification"] = df["gender"].apply(Gender_Classification)

print(df["Gender_Classification"].value_counts())
plt.figure(figsize = (10, 5))
sns.countplot(x='Gender_Classification', data=df)
plt.xlabel('Gender')
plt.ylabel('Count of Gender')
plt.title( 'Gender Classification');


# In[170]:


def SeniorCitizen(sc):
    if sc == 1:
        return "Senior Citizen"
    else:
        return "Junior Citizen"

print(df["SeniorCitizen"].value_counts())    
df['sc'] = df['SeniorCitizen'].apply(SeniorCitizen)

plt.figure(figsize = (10, 5))
sns.countplot(x='sc', data=df)
plt.xlabel('Category of Citizen')
plt.ylabel('Count of Citizen Category')
plt.title( 'Total Number of Citizen');


# In[172]:


# Convert the "InternetService" column to string

df['InternetService'] = df['InternetService'].astype(str)

print(df["InternetService"].value_counts())
df['InternetService'].hist(bins=20,)
plt.title( 'Internet Services');


# In[38]:


# Subplot Visualization of Tenure, Partner, Monthly Charges, Phone Services, Streaming Movies, Dependents.

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

plt1 = sns.countplot(x='tenure', data=df, ax=axs[0, 0])
plt1.set_title('Distribution of Tenure')

plt2 = sns.countplot(x='Partner', data=df, ax=axs[0, 1])
plt2.set_title('Distribution of Partner')

plt3 = sns.histplot(df['MonthlyCharges'], ax=axs[0, 2])
plt3.set_title('Distribution of Monthly Charges')

plt4 = sns.countplot(x='PhoneService', data=df, ax=axs[1, 0])
plt4.set_title('Distribution of Phone Service')

plt5 = sns.histplot(x='StreamingMovies', data=df, ax=axs[1, 1])
plt5.set_title('Distribution of Streaming Movies')

plt6 = sns.countplot(x='Dependents', data=df, ax=axs[1, 2])
plt6.set_title('Distribution of Dependents')

plt.tight_layout()
plt.show()


# ### Bivariate Analysis

# In[198]:


# Investigating relationship between Gender and Churn

# Bar plot using seaborn
sns.countplot(x='gender', hue='Churn', data=df)

# Set plot labels and title
plt.xlabel('gender')
plt.ylabel('Count')
plt.title('Churn by gender')

# Show the plot
plt.show()


# A bar plot visualizing the Churn rates between male and female customers with variable of interest as Churn and Gender indicates a lower churn rate among both gender within the telecom customer dataset. 
# 
# it is observed that the churn rates, representing the proportion of customers who discontinued services, differ slightly between genders. while, the Churn rate is low in both gender, the Non-Churn rate is high.

# In[204]:


# Investigating relationship between Tenure and Churn

# Bar plot using seaborn
sns.countplot(x='Tenure_Segment', hue='Churn', data=df)

print(df["Tenure_Segment"].value_counts())

# Set plot labels and title
plt.xlabel('Tenure_Segment')
plt.ylabel('Count')
plt.title('Churn Distribution across Tenure_Segment')

# Show the plot
plt.show()


# Churn Distribution Across Tenure Segments:
# 
# The bar plot illustrates the distribution of churn within different tenure segments.Noticeable differences in the count of churners and non-churners across tenure segments can be observed.
# 
# it is observed that the New tenure segments exhibit higher churn rates compared to others. it is observed that the longer the tenure the lower the Churn and the lower the tenure the higher the Churn (see analysis below).
# 
# T_S	       Non-Churn	Churn
# New	           58%	     42%
# Intermediate	78%	     22%
# LongTerm	   89%	     11%
# 
# The New tenure segment may require targeted retention efforts.
# 
# In conclusion, our analysis underscores the pivotal role of tenure in understanding and predicting customer churn. As tenure increases, the likelihood of churn decreases, emphasizing the importance of fostering customer loyalty, especially during the early stages of the customer relationship.

# In[4]:


df.head()


# In[37]:


# Bivariate analysis - Pairplot
plt.figure(figsize=(6,4))
columns_to_check = ["SeniorCitizen", "Partner", "Tenure_Segment", "MultipleLines", "StreamingTV", "MonthlyCharges", "Churn"]
sns.pairplot(df[columns_to_check], hue="Churn")
plt.show()


# ##### Multivariate Analysis

# In[34]:


# Multivariate analysis - Correlation Heatmap

correlation_table = df[columns_to_check].corr()

sns.heatmap(correlation_table, annot=True)
plt.show()


# In[36]:


df.head()


# ### Future Engineering/ Data Pre-Processing

# In[38]:


df.head()


# In[71]:


df1 = df[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",  "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]]

label = df[['Churn']]


# In[42]:


df.head()


# In[45]:


df1.dtypes


# ### Machine Learning

# In[79]:


# split the data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1, label, test_size=0.2, random_state=42)


# In[80]:


# Model Building

# Logistic Regression
lng = LogisticRegression()
lng.fit(X_train, y_train)
ly_pred = lng.predict(X_test)

print('Logistic Regression')
print('Accuracy:', accuracy_score(y_test,ly_pred))
print('Precision:', precision_score(y_test,ly_pred))
print('Recall:', recall_score(y_test,ly_pred))
print('F1-score:', f1_score(y_test,ly_pred))
print('AUC-ROC:', roc_auc_score(y_test,ly_pred))


# In[111]:


X_test


# In[75]:


ly_pred


# In[110]:


y_test


# In[82]:


# Confusion Matrix

lcm = confusion_matrix(ly_pred, y_test)

# Visualization in Confusion Matrix
sns.heatmap(lcm, annot=True,cmap = 'Blues', fmt = 'g')
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[135]:


# Random Forest Classifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rcm_pred = rfc.predict(X_test)

print('Random Forest Classifier')
print('Accuracy:', accuracy_score(y_test, rcm_pred))
print('Precision:', precision_score(y_test, rcm_pred))
print('Recall:', recall_score(y_test, rcm_pred))
print('F1-score:', f1_score(y_test, rcm_pred))
print('AUC-ROC:', roc_auc_score(y_test, rcm_pred))


# In[83]:


# Random Forest Classifier

rfc = confusion_matrix(y_test,rcm_pred)

# Visualize the confusion matrix
sns.heatmap(rfc,annot=True,cmap='Blues',fmt='g')
plt.xlabel('predicted')
plt.ylabel('actual');


# In[149]:


# Decision Tree

Dtc = DecisionTreeClassifier()
Dtc.fit(X_train, y_train)
y_pred = Dtc.predict(X_test)

print('Decision Tree')
print('Accuracy:', accuracy_score(y_test,y_pred))
print('Precision:', precision_score(y_test,y_pred))
print('Recall:', recall_score(y_test,y_pred))
print('F1-score:', f1_score(y_test,y_pred))
print('AUC-ROC:', roc_auc_score(y_test,y_pred))


# In[150]:


dtc = confusion_matrix(y_test,y_pred)

# Visualize the confusion matrix
sns.heatmap(dtc,annot=True,cmap='Blues',fmt='g')
plt.xlabel('predicted')
plt.ylabel('actual');


# In[141]:


# SGDClassifier

SDG_Classifier = SGDClassifier()
SDG_Classifier.fit(X_train, y_train)
ly_pred = SDG_Classifier.predict(X_test)

print('SGDClassifier')
print('Accuracy:', accuracy_score(y_test,ly_pred))
print('Precision:', precision_score(y_test,ly_pred))
print('Recall:', recall_score(y_test,ly_pred))
print('F1-score:', f1_score(y_test,ly_pred))
print('AUC-ROC:', roc_auc_score(y_test,ly_pred))


# In[142]:


SDG_Classifier = confusion_matrix(y_test,ly_pred)

# Visualize the confusion matrix
sns.heatmap(SDG_Classifier,annot=True,cmap='Blues',fmt='g')
plt.xlabel('predicted')
plt.ylabel('actual');


# In[151]:


# GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)

print('GradientBoostingClassifier')
print('Accuracy:', accuracy_score(y_test,gbc_pred))
print('Precision:', precision_score(y_test,gbc_pred))
print('Recall:', recall_score(y_test,gbc_pred))
print('F1-score:', f1_score(y_test,gbc_pred))
print('AUC-ROC:', roc_auc_score(y_test,gbc_pred))


# In[152]:


gbc = confusion_matrix(y_test,gbc_pred)

# Visualize the confusion matrix
sns.heatmap(gbc,annot=True,cmap='Blues',fmt='g')
plt.xlabel('predicted')
plt.ylabel('actual');


# In[207]:


# SVC

Svo = SVC()
Svo.fit(X_train, y_train)
Svo_pred = Svo.predict(X_test)

print('SVC')
print('Accuracy:', accuracy_score(y_test,Svo_pred))
print('Precision:', precision_score(y_test,Svo_pred))
print('Recall:', recall_score(y_test,Svo_pred))
print('F1-score:', f1_score(y_test,Svo_pred))
print('AUC-ROC:', roc_auc_score(y_test,Svo_pred))


# In[153]:


Svo = confusion_matrix(y_test,Svo_pred)

# Visualize the confusion matrix
sns.heatmap(gbc,annot=True,cmap='Blues',fmt='g')
plt.xlabel('predicted')
plt.ylabel('actual');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




