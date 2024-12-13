import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv('Life Expectancy Data.csv')

df.head()

df.shape

df.info()

df.isnull().sum()

df.duplicated().sum()

df.describe()

#to understand the distribution
for i in df.select_dtypes(include='number').columns:
  sns.histplot(data=df,x=i)
  plt.show()

#to find the outliers
for i in df.select_dtypes(include='number').columns:
  sns.boxplot(data=df,x=i)
  plt.show()

#to find the relationship
df.select_dtypes(include='number').columns

for i in['Year', 'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']:
       sns.scatterplot(data=df,y='Life expectancy ',x=i)
       plt.show()

#to find the correlation between the datas
s=df.select_dtypes(include='number').columns
plt.figure(figsize=(15,15))
sns.heatmap(df[s].corr(),annot=True)

#missing value treatment
for i in [' BMI ','Polio','Income composition of resources']: # Added a space before and after BMI
  df[i].fillna(df[i].median(),inplace=True)

df.isnull().sum()

#using knnimpute
from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=5)
for i in df.select_dtypes(include='number').columns:
  df[i]=imputer.fit_transform(df[[i]])

df.isnull().sum()

#outliers treatment
def wisker(col):
  q1,q3=np.percentile(col,[25,75])
  iqr=q3-q1
  lower_bound=q1-(1.5*iqr)
  upper_bound=q3+(1.5*iqr)
  return lower_bound,upper_bound
for i in ['GDP','Total expenditure',' thinness  1-19 years']:
  lower_bound,upper_bound=wisker(df[i])
  df[i]=np.where(df[i]>upper_bound,upper_bound,df[i])
  df[i]=np.where(df[i]<lower_bound,lower_bound,df[i])

#boxplot after removing outliers
for i in ['GDP','Total expenditure',' thinness  1-19 years']:
  sns.boxplot(data=df,x=i)
  plt.show()

#to save processed data-set into a new data-set
processed=df.to_csv('processed_dataset(led).csv',index=False)

df2=pd.read_csv('processed_dataset(led).csv')

df2.info()

#converting categorical to numerical values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df2.select_dtypes(include='object').columns:
  df2[i]=le.fit_transform(df2[i])

df2.info()

#normalization
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
for i in df2.select_dtypes(include='number').columns:
  df2[i]=scaler.fit_transform(df2[[i]])

df2.head()

#principle component analysis for dimensionality reduction
from sklearn.decomposition import PCA
pca=PCA(n_components=5)
df2=pca.fit_transform(df2)

df2.shape

#splitting datasets into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df2,df['Life expectancy '],test_size=0.2,random_state=42)

#printing test and train dataset

print("Training feature set shape:", x_train.shape)
print("Testing feature set shape:", x_test.shape)
print("Training target set shape:", y_train.shape)
print("Testing target set shape:", y_test.shape)





def evaluate_regression_models(x_train, x_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestRegressor(),
        "SVM": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Linear Regression": LinearRegression()
    }

    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "Mean Squared Error": mse,
            "R^2 Score": r2
        }

    return results


regression_results = evaluate_regression_models(x_train, x_test, y_train, y_test)

print("Regression Results:")
for model_name, metrics in regression_results.items():
    print(f"\n{model_name}:\n")
    print("Mean Squared Error:", metrics["Mean Squared Error"])
    print("R^2 Score:", metrics["R^2 Score"])
