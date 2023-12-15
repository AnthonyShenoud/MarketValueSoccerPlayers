from asyncio.events import BaseDefaultEventLoopPolicy
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from multiprocessing.sharedctypes import Value
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import patsy

# ------- Read in players data from excel parsed from parsing.py -------
df = pd.read_csv('top5_leagues_all_players.csv')

# display options when printing data to the console
#pd.set_option('display.max_columns', 15)
#pd.set_option('display.max_rows', 15)

# ------- logic to convert currencies and values into the same units ------- 
i = 0
temp_values = []
for value in df['Value']:
    input = value[1:]
    if 'M' in value:
        temp_values.append(float(input.replace('M', '')) * 1000000)
    elif 'K' in value:
        temp_values.append(float(input.replace('K', '')) * 1000)
    else:
        temp_values.append(float(input))
    i+=1

df['Value'] = temp_values
count = 0 
for value in df:
    temp_values = []
    for j in df[value]:
        if '-' in str(j):
            index = j.find('-')
            temp_values.append(j[index+1:])
        elif '+' in str(j):
            temp_values.append(j[:index-1])
            count+=1
        else:
            temp_values.append(j)
    df[value] = temp_values

# ------- print statement to display a players main 6 FIFA attributes ------- 
#print(df.loc[[1],['name','Value', 'Pace / Diving', 'Shooting / Handling', 'Passing / Kicking', 'Dribbling / Reflexes', 'Defending / Pace', 'Physical / Positioning']])
# ------- drop columns that are not correlated to salary and non-numeric and duplicate rows ------- 
df.drop(columns=['name', "Height", "Weight", "foot", "Team & Contract", "Wage", "Release clause", "ID", "Best position"], inplace=True)
df = df.drop_duplicates()

# ------- create correlation matrix ------- 
correlation_matrix = df.corr()
sorted_correlation_matrix = correlation_matrix['Value'].sort_values(ascending=False)
# Create a list of tuples with variable name and correlation value pairs
sorted_correlation_list = [(variable, correlation) for variable, correlation in sorted_correlation_matrix.items()]

# ------- print statement to display correlation between attributes and value ------- 
#for variable, correlation in sorted_correlation_list:
#    print(f"{variable}: {correlation}")

# ------- 6 selected attributes encompassing mental, physical, and skills characteristics ------- 
selected_columns = ['Value', 'Pace / Diving', 'Shooting / Handling', 'Passing / Kicking', 'Dribbling / Reflexes', 'Defending / Pace', 'Physical / Positioning']
six_attr = ['Pace / Diving', 'Shooting / Handling', 'Passing / Kicking', 'Dribbling / Reflexes', 'Defending / Pace', 'Physical / Positioning']
selected_df = df[selected_columns].copy()

# ------- IQR Outlier Detection ------- 
temp_iqr_df = selected_df.copy()
Q1 = temp_iqr_df.quantile(0.25)
Q3 = temp_iqr_df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_df = temp_iqr_df[~((temp_iqr_df < lower_bound) | (temp_iqr_df > upper_bound)).any(axis=1)]

# ------- Percentile Method with Winsorization Outlier Detection ------- 
percentile_df = df.copy()
upper_limit = percentile_df['Value'].quantile(0.99)
lower_limit = percentile_df['Value'].quantile(0.01)
percentile_df = percentile_df[(percentile_df['Value'] <= upper_limit) & (percentile_df['Value'] >= lower_limit)]
percentile_df['Value'] = np.where(percentile_df['Value'] >= upper_limit,
        upper_limit,
        np.where(percentile_df['Value'] <= lower_limit,
        lower_limit,
        percentile_df['Value']))

# ------- Create a boxplot with the dataframe -------
def create_boxplot(box_df):
    fig = plt.figure(figsize=[30,25])
    sns.set(style='white', font_scale=1.4)
    data = box_df[six_attr]
    sns.boxplot(data=data)
    plt.grid(False)
    plt.xlabel('6 Attributes', fontsize=20, weight='bold')
    plt.ylabel('Ratings', fontsize=20, weight='bold')
    plt.title('Distribution of Ratings', fontsize=20, weight='bold')
    sns.despine()
    plt.show()

# ------- Value vs 6 attributes plots from iqr dataframe -------
sns.set(style='white',font_scale=0.5)
sns.pairplot(iqr_df, x_vars='Value')
#plt.show()

# ------- Dataframe for logarithmic regression -------
iqr_log_df = percentile_df.copy()
mask = percentile_df['Value'] > 0  
iqr_log_df['Value'] = np.log(percentile_df.loc[mask, 'Value']) 
iqr_log_df = iqr_log_df.rename(columns={'Value':'Log Market Value'})
sns.set(style='white',font_scale=0.5)
sns.pairplot(iqr_log_df, x_vars='Log Market Value')
#plt.show()


# ------- Create a statistical model where the Value is depenedent on the 6 attributes -------
y, X = patsy.dmatrices("Q('Value') ~ Q('Pace / Diving') + Q('Shooting / Handling') + Q('Passing / Kicking') + Q('Dribbling / Reflexes') + Q('Defending / Pace') + Q('Physical / Positioning')", data=iqr_df, return_type="dataframe")
model = sm.OLS(y, X)
fit = model.fit()
#print(fit.summary())

# ------- Create a statistical model where the Value is depenedent on the 6 attributes but with the logarithmic dataframe -------
y_log, X_log = patsy.dmatrices("Q('Log Market Value') ~ Q('Pace / Diving') + Q('Shooting / Handling') + Q('Passing / Kicking') + Q('Dribbling / Reflexes') + Q('Defending / Pace') + Q('Physical / Positioning')", data=iqr_log_df, return_type="dataframe")
model = sm.OLS(y_log, X_log)
fit = model.fit()
#print(fit.summary())

def create_combined_df(X,y):
    combined_df = pd.concat([X, y], axis=1)
    combined_df.dropna(inplace=True)
    return combined_df

def diagnostic_plot(x, y):
    combined_df = create_combined_df(x,y)
    x = combined_df.iloc[:, :-1]
    y = combined_df.iloc[:, -1]

    plt.figure(figsize=(20,5))
    
    rgr = LinearRegression()
    rgr.fit(x,y)
    pred = rgr.predict(x)

    plt.subplot(1, 3, 1)
    plt.scatter(pred,y,alpha=0.1)
    plt.plot(y, y, color='red',linewidth=1,)
    plt.title("Regression fit")
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    
    plt.subplot(1, 3, 2)
    res = y - pred
    plt.scatter(pred, res,alpha=0.1)
    plt.title("Residual plot")
    plt.xlabel("Prediction")
    plt.ylabel("Residuals (Observed - Predicted)")
    
    plt.subplot(1, 3, 3)
    stats.probplot(res, dist="norm", plot=plt,)
    plt.title("Normal Q-Q plot")
    plt.show()

def regressionTechniques(X,y):
    combined_df = create_combined_df(X,y)
    X = combined_df.iloc[:, :-1]
    y = combined_df.iloc[:, -1]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    reg = LinearRegression()
    reg.fit(X_train_val, y_train_val)
    model = reg.fit(X,y)
    print('')
    print('Score for linear regression: ' + str(reg.score(X_train_val, y_train_val)))

    forest = RandomForestRegressor()
    forest.fit(X_train_val, y_train_val)
    print('Score for random forest: ' + str(forest.score(X_train_val, y_train_val)))
    print('')
    predictions_forest = forest.predict(X_test)
    predictions_reg = reg.predict(X_test)
# Create a new DataFrame for predictions and actual values
    results_df = pd.DataFrame({'Predicted Market Value (Random Forest)': predictions_forest,
                           'Predicted Market Value (Linear Regression)': predictions_reg,
                           'Actual Market Value': y_test})
    print(results_df)

    mean_sq_err_reg = mean_squared_error(y_test, predictions_reg)
    print('')
    print('Mean Square Error for linear regression: ' + str(mean_sq_err_reg))
    mean_sq_err_forest = mean_squared_error(y_test, predictions_forest)

    print('Mean Square Error for random forrest: ' + str(mean_sq_err_forest))


#regressionTechniques(X,y)
regressionTechniques(X_log, y_log)

#diagnostic_plot(X,y)
#diagnostic_plot(X_log, y_log)