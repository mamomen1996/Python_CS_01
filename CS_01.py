#!/usr/bin/env python
# coding: utf-8

# # Case-Study Title: Simple Linear Regression Analysis
# ###### Data Analysis methodology: CRISP-DM
# ###### Dataset: Toyota Used Cars certified features and dealing (sold) prices in Europe
# ###### Case Goal: Price Recommendation Intelligence System for Toyota Used Cars in Europe Trading Platform

# # Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# # Read Data from File

# In[2]:


data = pd.read_csv('CS_01.csv')


# In[3]:


data.shape  # 1325 records, 10 variables


# # Business Understanding
#  * know business process and issues
#  * know the context of the problem
#  * know the order of numbers in the business

# # Data Understanding
# ## Data Inspection (Data Understanding from Free Perspective)
# ### Dataset variables definition

# In[4]:


data.columns


# * **Price**         : Sales (sold) price in Euro      -> what we want to predict
# * **Age**           : Age of a used car in month 
# * **KM**            : Kilometerage usage
# * **FuelTyp**       : Petrol, Diesel, CNG             -> Categorical (factor)
# * **HP**            : Horse power     
# * **MetColor**      : 1 : if Metallic color, 0 : Not  -> Categorical (factor)
# * **Automatic**     : 1 : if Automatic, 0 : Not       -> Categorical (factor)
# * **CC**            : Engine displacement in cc
# * **Doors**         : # of doors                      -> Categorical (factor)
# * **Weight**        : Weight in Kilogram

# ## Data Exploring (Data Understanding from Statistical Perspective)
# ### Overview of Dataframe

# In[5]:


type(data)


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[10]:


# Do we have any NA in our Variables?
data.isna().sum()

# We have no MV problem in this dataset


# In[11]:


# Check for abnormality in data
data.describe(include='all')


# ### Categorical variables should be stored as factor

# In[27]:


data.FuelType = data.FuelType.values.astype(str)
data.MetColor = data.MetColor.values.astype(bool)
data.Automatic = data.Automatic.values.astype(bool)
data.Doors = data.Doors.values.astype(str)


# In[28]:


data.describe(include='all')


# ### Univariate Profiling (check each variable individually)
# #### Categorical variables
# Check to sure that have good car distribution in each category
# 
# **Rule of Thumb**: we must have atleast 30 observation in each category

# In[32]:


data.FuelType.value_counts()

# CNG cars sample size is very small -> 17/1325 < 0.05


# > We have few data|sample in **CNG** category of **FuelType** -> it can affect on price prediction of this cars category

# In[33]:


data.MetColor.value_counts()


# In[34]:


data.Automatic.value_counts()


# > We have few data|sample in **True** category of **Automatic** -> it can affect on price prediction of this cars category

# In[35]:


data.Doors.value_counts()

# 2-Doors cars sample size is very small -> 2/1325 < 0.05


# In[37]:


data.loc[data.Doors == '2']  # abnormality (error in data recording process)


# #### Continuous variables
# distribution: plot Histogram

# In[30]:


var_ind = [0, 1, 2, 4, 7, 9]
plot = plt.figure(figsize = (12, 6))
plot.subplots_adjust(hspace = 0.5, wspace = 0.5)
for i in range(1, 7):
    a = plot.add_subplot(2, 3, i)
    a.hist(data.iloc[:, var_ind[i - 1]], alpha = 0.7)
    a.title.set_text('Histogram of ' + data.columns[var_ind[i - 1]])


# In[31]:


# Box plot of Price
plt.boxplot(data['Price'], showmeans = True)
plt.title('Boxplot of Price')

# Price is skewed to right a little bit


# ### Bivariate Profiling (measure 2-2 relationships between variables)
# #### Two Continuous variables (Correlation Analysis)

# In[38]:


data[['Price', 'KM']].corr(method = 'pearson')  # high correlation for this context (Used Car price)


# In[39]:


# correlation table between Price and continuous variables
corr_table = round(data[['Price','Age','KM','HP','CC','Weight']].corr(method = 'pearson'), 2)
corr_table   # choose continuous variables which have high corr with price and consider them as feature in regression model (which variable is important for price prediction)


# > **CC** has very small corr with **Price**, so it can not be good predictor in modeling
# 
# > **Price** has high correlation with **Age** and **KM**
# 
# > **Weight** and **CC** have high correlation with each other

# In[40]:


sns.heatmap(corr_table, annot = True)


# Multicollinearity (having high correlation between predictor variables):
# 
# abs(corr) >= 0.30: Multicollinearity problem danger!
# 
# * **Weight** has 0.66 corr with **CC**
# * **KM** has 0.39 corr with **Age**
# * **KM** has 0.39 corr with **CC**
# * **KM** has -0.33 corr with **HP**

# In[41]:


# Scatter Plot (between Price and other continuous variables 2 by 2)
var_ind = [1, 2, 4, 7, 9]
plot = plt.figure(figsize = (12, 6))
plot.subplots_adjust(hspace = 0.5, wspace = 0.5)
for i in range(1, 6):
    a = plot.add_subplot(2, 3, i)
    a.scatter(x = data.iloc[:, var_ind[i - 1]],
             y = data.iloc[:, 0],
             alpha = 0.5)
    a.title.set_text('Price vs. ' + data.columns[var_ind[i - 1]])


# > **Age** and **Price** have strong linear relationship
# 
# > **CC** and **HP** are categorical like!

# # Data PreProcessing
# ## Divide Dataset into Train and Test randomly
# * Learn model in Train dataset
# * Evaluate model performance in Test dataset

# In[42]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3, random_state = 123)

# according to the dataset size: 70% - 30% 


# In[43]:


# train data distribution must be similar to test data distribution
train.shape


# In[44]:


train.describe()


# In[45]:


test.shape


# In[46]:


test.describe()


# # Modeling
# ## Model 1: Simple Linear Regression (Univariate Regression)
# based-on previous analysis, it seems that KM is important to explain Price variance (corr = -52%)

# In[47]:


# Price vs. KM

# Define the features set X
X = train['KM']
X = sm.add_constant(X)  # adding a constant (a column of 1)

# Define response variable
y = train['Price']


# In[48]:


X.head()


# In[49]:


y.head()


# In[50]:


# Regress Price on KM
m1 = sm.OLS(y, X).fit()
m1.summary()  # results of m1 regression model


# > **R-squared = 0.277**: 28% of **Price** variance has been explained by **KM**
# 
# > Consider the problem context, for price prediction, R-squared = 0.28 is not good model, we need 0.70

# In[57]:


# Plot Regression Line with Confidence-Interval
sns.regplot(x = 'KM', 
            y = 'Price',
            data = data, 
            scatter_kws={'color':'blue', 'alpha':0.5},
            line_kws={'color':'red'})

# variance of 'Price' based-on 'KM' is high around regression line


# **Main Question**: can we generalize this line to population? -> F-test and then t-test
# 
# Check Assumptions of Regression:
# 
# 1. Normality of residuals (Errors)

# In[58]:


m1.resid  # errors of model


# In[61]:


# Plot Histogram of residuals
sns.histplot(m1.resid, stat = 'probability', 
            kde = True,
            alpha = 0.7, color = 'green',
            bins = 20)

# skewed to right (have a tail along right)


# In[62]:


# QQ-plot
qqplot_m1 = sm.qqplot(m1.resid, line = 's')
plt.show()

# we have serious deviations from normal distribution


# In[63]:


# Jarque-Bera Test (Normal Skewness = 0)
  # H0: the data is normally distributed
  # if p-value < 0.05, then reject normality assumption

# Omnibus K-squared normality test (Normal Kurtosis = 3)
  # H0: the data is normally distributed
  # if p-value < 0.05, then reject normality assumption

print(m1.summary())


# In[64]:


# Shapiro-Wilk Test for Normality (instead of Skewness and Kurtosis Test)
  # H0: the data is normally distributed
  # if p-value < 0.05, then reject normality assumption

from scipy.stats import shapiro
shapiro_m1 = shapiro(m1.resid)
shapiro_m1


# > **result**: Residuals are not Normally Distributed -> reject first Assumption of Regression
# 
# 2. Residuals independency

# In[65]:


# Diagnostic plot for checking Heroscedasticity problem

sns.regplot(x = m1.fittedvalues,
            y = m1.resid,
            lowess = True,
            scatter_kws = {'color': 'black'},
            line_kws = {'color': 'red'})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()

# Top three observations with the greatest absolute value of the residual
top3 = abs(m1.resid).sort_values(ascending = False)[:3]
for i in top3.index:
    plt.annotate(i, xy = (m1.fittedvalues[i], m1.resid[i]), color = 'red')


# > **result**: We see Heteroscedasticity problem in model (variance of residuals is not constant)

# In[66]:


# Check Cook's distance
m1.get_influence().summary_frame().cooks_d  # extract Cook's distance of every observation


# In[70]:


# if we have a observation with Cook's Distance > 1, that makes a problem
sum(m1.get_influence().summary_frame().cooks_d > 1)


# > **result**: there is no Cook's Distance > 1
# 
# Our m1 model problems:
#  1. has Heteroscedasticity problem
#  2. Errors are not normally distributed
#  
# > So, this model has problem. and t-test results of it are not reliable yet!
#  
# ## Model 2: Quadratic Regression (Multivariate Regression)

# In[71]:


plt.scatter(x = data['KM'], y = data['Price'])


# > it seems that the relationship between these two variables in this sample and this data-range isn't linear, it is non-linear relationship 
# 
# **Hypothesis**: there is a non-linear relationship between **Price** and **KM** -> fit a 2-degree curve to describe it

# In[72]:


train['KM_2'] = train['KM'] ** 2  # create new variable
train.head()


# In[73]:


# Define the feature set X
X = train[['KM', 'KM_2']]
X = sm.add_constant(X)  # adding a constant

# Define response variable
y = train['Price']


# In[74]:


X.head()


# In[75]:


# Regression model
m2 = sm.OLS(y, X).fit()
m2.summary()


# Check Assumptions of Regression
# 
# 1. Normality of residuals (Errors)

# In[76]:


# Plot Histogram of residuals
sns.histplot(m2.resid, stat = 'probability',
             kde = True, alpha = 0.7, color = 'green',
             bins = 20)

# skewed to right


# In[77]:


# QQ-plot
qqplot_m2 = sm.qqplot(m2.resid, line = 's')
plt.show()


# In[78]:


# Jarque-Bera Test (Skewness = 0 ?)
  # H0: the data is normally distributed
  # p-value < 0.05 reject normality assumption

# Omnibus K-squared normality test
  # H0: the data is normally distributed
  # p-value < 0.05 reject normality assumption

print(m2.summary())


# > **result**: Residuals are not Normally Distributed -> reject first Assumption of Regression
# 
# 2. Residuals independency

# In[79]:


# Diagnostic plot
sns.regplot(x = m2.fittedvalues, y = m2.resid, lowess = True,
               scatter_kws = {'color': 'black'}, line_kws = {'color': 'red'})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()

# Top three observations with the greatest absolute value of the residual
top3 = abs(m2.resid).sort_values(ascending = False)[:3]
for i in top3.index:
    plt.annotate(i, xy = (m2.fittedvalues[i], m2.resid[i]), color = 'red')


# In[80]:


# Check Cook's distance
sum(m1.get_influence().summary_frame().cooks_d > 1)  # are any observation with Cook's Distance > 1


# Linear Regression vs Quadratic Regression

# In[87]:


plt.scatter(x = train['KM'], y = train['Price'], alpha = 0.6)

# fit Linear (1-degree) Regression on data
params1 = np.polyfit(train['KM'], train['Price'], 1)  # fit 1-degree
xp = np.linspace(train['KM'].min(), train['KM'].max(), 100)  # generate 100 continuous number on X axis
yp1 = np.polyval(params1, xp)  # generate equivalent yp for xp
plt.plot(xp, yp1, alpha = 0.9, linewidth = 2, color = 'green', label = 'Linear Regression')

# fit Quadratic (2-degree) Regression on data
params2 = np.polyfit(train['KM'], train['Price'], 2)  # fit 2-degree
yp2 = np.polyval(params2, xp)  # generate equivalent yp for xp
plt.plot(xp, yp2, alpha = 0.9, linewidth = 2, color = 'red', label = 'Quadratic Regression')

plt.xlabel('KM', fontsize = 12)
plt.ylabel('Price', fontsize = 12)
plt.title('Price vs. KM', fontsize = 12)
plt.legend()


# Check having Multicollinearity problem via VIF

# In[88]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):  # X: features matrix
    vif = pd.DataFrame()
    vif['variables'] = X.columns  # column names
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


# In[89]:


calc_vif(X)  # calculate VIF for each variable (if VIF > 10 then Multicollinearity problem is serious)


# > We have strong Multicollinearity here because we define KM_2 variable based on KM variable

# In[90]:


# Scale variable -> solve Multicollinearity problem
train['KM_scaled'] = (train['KM'] - train['KM'].mean()) / train['KM'].std()
train['KM_scaled_2'] = train['KM_scaled'] ** 2
train.head()


# In[91]:


# Define the feature set X
X = train[['KM_scaled', 'KM_scaled_2']]
X = sm.add_constant(X)

# Define response variable
y = train['Price']


# In[92]:


# Regression model
m2_2 = sm.OLS(y, X).fit()
m2_2.summary()


# In[93]:


# Plot Histogram of residuals
sns.histplot(m2_2.resid, stat = 'probability',
             kde = True, alpha = 0.7, color = 'green',
             bins = 20)


# In[94]:


calc_vif(X)


# ## Model 3: Use All Variables

# In[96]:


import statsmodels.formula.api as smf

# Regress 'Price' on all other predictor variables
m3 = smf.ols(
    formula = 'Price ~ KM_scaled + KM_scaled_2 + Age + C(FuelType) + HP + C(MetColor) + C(Automatic) + CC + C(Doors) + Weight',
    data = train).fit()

m3.summary()


# Remove in-significant variables (consider the t-test results)

# In[97]:


#Removing variables: MetColor
train.groupby(['MetColor'])['Price'].mean()  # calculate mean(Price) for each Category


# In[98]:


# Boxplot for Price vs. MetColor (Descriptive Analysis)
sns.boxplot(x = 'MetColor', y = 'Price', data = train, showmeans = True)


# > **MetColor** is not useful predictor for **Price** (has not significant effect on it)

# In[99]:


#Removing variables: Doors
train.groupby(['Doors'])['Price'].mean()  # calculate mean(Price) for each Category


# In[100]:


# Boxplot for Price vs. Doors (Descriptive Analysis)
sns.boxplot(x = 'Doors', y = 'Price', data = train, showmeans = True)


# > **Doors** is not useful predictor for **Price** (has not significant effect on it)

# In[101]:


#Removing variables: HP
train[['Price', 'HP']].corr()


# In[102]:


# Scatter plot for Price vs. HP
plt.scatter(x = train['HP'], y = train['Price'])
plt.title('Price vs. HP')
plt.xlabel('HP')
plt.ylabel('Price')


# > **HP** is not useful predictor for **Price** (has not significant effect on it)

# In[103]:


m3 = smf.ols(formula = 'Price ~ KM_scaled + KM_scaled_2 + Age + C(FuelType) + C(Automatic) + CC + Weight',
            data = train).fit()

m3.summary()


# Check Assumptions of Regression
# 
# 1. Normality of residuals

# In[110]:


# Plot Histogram of residuals
sns.histplot(m3.resid, stat = 'probability',
             kde = True, alpha = 0.7, color = 'green',
             bins = 20)

# skewed to left


# In[111]:


# QQ-plot
qqplot_m3 = sm.qqplot(m3.resid, line = 's')
plt.show()


# ## Model 4: Improved Multiple Regression
# simplify **FuelType** variable to binary **IfPetrol** variable:

# In[112]:


train.loc[train['FuelType'] == 'Petrol', 'IfPetrol'] = True
train.loc[train['FuelType'] != 'Petrol', 'IfPetrol'] = False
train.head()


# In[113]:


# Regression model
m4 = smf.ols(formula = 'Price ~ KM_scaled + KM_scaled_2 + Age + C(IfPetrol) + C(Automatic) + CC + Weight',
            data = train).fit()

m4.summary()


# Check Assumptions of Regression
# 
# 1. Normality of residuals

# In[114]:


# Plot Histogram of residuals
sns.histplot(m4.resid, stat = 'probability',
            kde = True, alpha = 0.7, color = 'green',
            bins = 25)

# skewed to left


# In[115]:


# QQ-plot
qqplot_m4 = sm.qqplot(m4.resid, line = 's')
plt.show()


# In[116]:


# Jarque-Bera Test (Skewness = 0 ?)
  # H0: the data is normally distributed
  # p-value < 0.05 reject normality assumption

# Omnibus K-squared normality test
  # H0: the data is normally distributed
  # p-value < 0.05 reject normality assumption

print(m4.summary())


# Remove some few outliers

# In[117]:


# Remove the outliers which have the most impact on the Regression line
sns.regplot(x = m4.fittedvalues, y = m4.resid, lowess = True,
               scatter_kws = {'color': 'black'}, line_kws = {'color': 'red'})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()

# Top three observations with the greatest absolute value of residual
top3 = abs(m4.resid).sort_values(ascending = False)[:3]
for i in top3.index:
    plt.annotate(i, xy = (m4.fittedvalues[i], m4.resid[i]), color = 'red')


# Observations with the greatest residual

# In[118]:


m4.resid.sort_values(ascending = False)[:5]  # first 5 positive residuals (top 5)


# In[119]:


m4.resid.sort_values(ascending = True)[:5]  # first 5 negative residuals (bottom 5)


# Remove Cases

# In[125]:


# remove 10 observations with the greatest absolute Errors
train2 = train.drop(index = [82, 111, 113, 283, 292, 446, 490, 585, 943, 947])


# In[126]:


train2.shape


# In[127]:


train[train.index == 490]  # see observation with index 490


# In[128]:


# Regression model
m4_2 = smf.ols(formula = 'Price ~ KM_scaled + KM_scaled_2 + Age + C(IfPetrol) + C(Automatic) + CC + Weight',
              data = train2).fit()

m4_2.summary()


# * Adj. R-squared improved: 83%
# * **Automatic** is not significant
# * Prob(JB) and Prob(Omnibus) are > 0.05 -> residuals are Normally distributed
# 
# Check Assumptions of Regression
# 1. Normality of residuals

# In[129]:


# Plot Histogram of residuals
sns.histplot(m4_2.resid, stat = 'probability',
            kde = True, alpha = 0.7, color = 'green',
            bins = 20)


# In[130]:


# QQ-plot
qqplot_m4_2 = sm.qqplot(m4_2.resid, line = 's')
plt.show()


# In[131]:


# Jarque-Bera Test (Skewness = 0 ?)
  # H0: the data is normally distributed
  # p-value < 0.05 reject normality assumption

# Omnibus K-squared normality test
  # H0: the data is normally distributed
  # p-value < 0.05 reject normality assumption

print(m4_2.summary())


# In[132]:


sns.regplot(x = m4_2.fittedvalues, y = m4_2.resid, lowess = True,
               scatter_kws = {'color': 'black'}, line_kws = {'color': 'red'})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()


# > Variance of residuals is almost constant
# 
# Check Cook's Distance

# In[133]:


sum(m4_2.get_influence().summary_frame().cooks_d > 1)


# > Regression Assumptions are confirmed -> we can consider to results of t-test
# 
# > **Automatic** is not significant (based on t-test)

# In[134]:


# Final Regression model: remove 'Automatic' variable
m4_2 = smf.ols(formula = 'Price ~ KM_scaled + KM_scaled_2 + Age + C(IfPetrol) + CC + Weight',
              data = train2).fit()

m4_2.summary()


# In[135]:


(train.shape[0] - train2.shape[0]) / train.shape[0] * 100


# > Number of removed observations from train is less than 2%

# # Model Evaluation
# Test the Model

# In[136]:


m4_2.params  # Coefficients of the model (Regression Coefficients)


# In[137]:


m4_2.conf_int(alpha = 0.05)  # Confidence Intervals for model parameters


# Data Preparation (test data)

# In[139]:


test['KM_scaled'] = (test['KM'] - test['KM'].mean()) / test['KM'].std()
test['KM_scaled_2'] = test['KM_scaled'] ** 2
test.loc[test['FuelType'] == 'Petrol', 'IfPetrol'] = True
test.loc[test['FuelType'] != 'Petrol', 'IfPetrol'] = False
test.head()


# Prediction on test

# In[140]:


test_pred = m4_2.predict(test)
test_pred  # prediction of 'Price' for each observation


# Actual vs. Prediction

# In[141]:


plt.scatter(x = test['Price'], y = test_pred)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(test['Price'].min(), test['Price'].max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# Absolute Error mean, median, sd, max, min

# In[142]:


abs_error = abs(test['Price'] - test_pred)
abs_error.describe()


# In[145]:


plt.hist(abs_error)


# In[146]:


sns.boxplot(y = abs_error)


# Absolute Error Percentage median, sd, mean, max, min

# In[147]:


e_percent = round(abs(test['Price'] - test_pred) / test['Price'] * 100, 2)
e_percent.describe()


# > **75%** of predictions have less than **12.59%** Absolute Error

# In[148]:


sum(e_percent <= 15) / len(e_percent) * 100


# > **84%** of our predictions have less than **15%** Absolute Error
