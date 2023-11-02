
# Importing Libraries

```python
# Basic packages
import pandas as pd
import numpy as np
import scipy as spy
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline
```


```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

---

# Data ingestion


```python
# Load dataset
data = pd.read_csv("InsuranceBusinessStats.csv")
```

---

## Preview dataset


```python
# Preview the dataset
# View the first 5, last 5 and random 10 rows
print('First five rows', '--'*55)
display(data.head())

print('Last five rows', '--'*55)
display(data.tail())

print('Random ten rows', '--'*55)
np.random.seed(1)

display(data.sample(n=10))
```

    First five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>


    Last five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>male</td>
      <td>30.97</td>
      <td>3</td>
      <td>no</td>
      <td>northwest</td>
      <td>10600.5483</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>female</td>
      <td>31.92</td>
      <td>0</td>
      <td>no</td>
      <td>northeast</td>
      <td>2205.9808</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>female</td>
      <td>36.85</td>
      <td>0</td>
      <td>no</td>
      <td>southeast</td>
      <td>1629.8335</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>female</td>
      <td>25.80</td>
      <td>0</td>
      <td>no</td>
      <td>southwest</td>
      <td>2007.9450</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>female</td>
      <td>29.07</td>
      <td>0</td>
      <td>yes</td>
      <td>northwest</td>
      <td>29141.3603</td>
    </tr>
  </tbody>
</table>
</div>


    Random ten rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>559</th>
      <td>19</td>
      <td>male</td>
      <td>35.530</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>1646.42970</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>57</td>
      <td>male</td>
      <td>31.540</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>11353.22760</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>51</td>
      <td>male</td>
      <td>37.000</td>
      <td>0</td>
      <td>no</td>
      <td>southwest</td>
      <td>8798.59300</td>
    </tr>
    <tr>
      <th>460</th>
      <td>49</td>
      <td>female</td>
      <td>36.630</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>10381.47870</td>
    </tr>
    <tr>
      <th>802</th>
      <td>21</td>
      <td>male</td>
      <td>22.300</td>
      <td>1</td>
      <td>no</td>
      <td>southwest</td>
      <td>2103.08000</td>
    </tr>
    <tr>
      <th>298</th>
      <td>31</td>
      <td>male</td>
      <td>34.390</td>
      <td>3</td>
      <td>yes</td>
      <td>northwest</td>
      <td>38746.35510</td>
    </tr>
    <tr>
      <th>481</th>
      <td>49</td>
      <td>male</td>
      <td>37.510</td>
      <td>2</td>
      <td>no</td>
      <td>southeast</td>
      <td>9304.70190</td>
    </tr>
    <tr>
      <th>616</th>
      <td>56</td>
      <td>female</td>
      <td>28.595</td>
      <td>0</td>
      <td>no</td>
      <td>northeast</td>
      <td>11658.11505</td>
    </tr>
    <tr>
      <th>763</th>
      <td>27</td>
      <td>male</td>
      <td>26.030</td>
      <td>0</td>
      <td>no</td>
      <td>northeast</td>
      <td>3070.80870</td>
    </tr>
    <tr>
      <th>750</th>
      <td>37</td>
      <td>female</td>
      <td>26.400</td>
      <td>0</td>
      <td>yes</td>
      <td>southeast</td>
      <td>19539.24300</td>
    </tr>
  </tbody>
</table>
</div>


**Initial observations**
* `sex`, `smoker` & `region` are categorical variables.
* `age`, `bmi`, `children` & `charges` are numerical variables.

----

## Variable List


```python
# Display list of variables in dataset
variable_list = data.columns.tolist()
print(variable_list)
```

    ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    

---

## Dataset shape


```python
shape = data.shape
n_rows = shape[0]
n_cols = shape[1]
print(f"The Dataframe consists of '{n_rows}' rows and '{n_cols}' columns")
```

    The Dataframe consists of '1338' rows and '7' columns
    

---

## Data info


```python
# Get info of the dataframe columns
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB
    

**Observations**
* All columns have values as there are `1338` rows and each column has `1338 non-null` elements
* `sex`, `smoker` & `region` have the **object** datatype. They should be categorical values.

---

**Panda Object Variable states**


```python
# Panda Object Variable states function

def pandas_object_states(data):
    """
    This function checks if the variable type is pandas Object and
    displays the states and counts of each
    """
    # Loop through all variables
    for var in data.columns:
        # Check for pandas Object type
        if data[var].dtypes == "object":
            print('Unique values in', var, 'are :')
            print(data[var].value_counts().sort_index())
            print('--'*55)
```


```python
# Check the states of all pandas Object variables
pandas_object_states(data)
```

    Unique values in sex are :
    sex
    female    662
    male      676
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in smoker are :
    smoker
    no     1064
    yes     274
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in region are :
    region
    northeast    324
    northwest    325
    southeast    364
    southwest    325
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    

---

**Convert Pandas Objects to Category type**


```python
# Convert variables with "object" type to "category" type
for i in data.columns:
    if data[i].dtypes == "object":
        data[i] = data[i].astype("category") 

# Confirm if there no variables with "object" type
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype   
    ---  ------    --------------  -----   
     0   age       1338 non-null   int64   
     1   sex       1338 non-null   category
     2   bmi       1338 non-null   float64 
     3   children  1338 non-null   int64   
     4   smoker    1338 non-null   category
     5   region    1338 non-null   category
     6   charges   1338 non-null   float64 
    dtypes: category(3), float64(2), int64(2)
    memory usage: 46.3 KB
    

**Missing value summary function**


```python
def missing_val_chk(data):
    """
    This function to checks for missing values 
    and generates a summary.
    """
    if data.isnull().sum().any() == True:
        # Number of missing in each column
        missing_vals = pd.DataFrame(data.isnull().sum().sort_values(
            ascending=False)).rename(columns={0: '# missing'})

        # Create a percentage missing
        missing_vals['percent'] = ((missing_vals['# missing'] / len(data)) *
                                   100).round(decimals=3)

        # Remove rows with 0
        missing_vals = missing_vals[missing_vals['# missing'] != 0].dropna()

        # display missing value dataframe
        print("The missing values summary")
        display(missing_vals)
    else:
        print("There are NO missing values in the dataset")
```

**Missing Values Check**


```python
#Applying the missing value summary function
missing_val_chk(data)
```

    There are NO missing values in the dataset
    

---

**Duplicate row check function**


```python
def df_duplicate_removal(data):
    """
    This function checks if there are any duplicated rows in the dataframe.
    If any, it displays the rows, keep the first occurence and drops the 
    duplicates.

    The new dataframe shape is calculated and returned
    """
    # Check if duplicated rows exist
    if any(data.duplicated() == True):
        print("The following is/are the duplicated row(s) in the dataframe")
        # Displays the duplicated row(s)
        display(data[data.duplicated() == True])
        # Drops the duplicates inplace while keeping the first occurence
        data.drop_duplicates(keep="first", inplace=True)

        # Check and returns the shape of the new dataframe
        new_df_shape = data.shape
        n_rows = new_df_shape[0]
        n_cols = new_df_shape[1]
        print(
            f"The new dataframe  consists of '{n_rows}' rows and '{n_cols}' columns")
    else:
        print("There is/are no duplicated row(s) in the dataframe")
```


```python
df_duplicate_removal(data)
```

    The following is/are the duplicated row(s) in the dataframe
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>581</th>
      <td>19</td>
      <td>male</td>
      <td>30.59</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>1639.5631</td>
    </tr>
  </tbody>
</table>
</div>


    The new dataframe  consists of '1337' rows and '7' columns
    

## 5 Point Summary

**Numerical type Summary**


```python
# Five point summary of all numerical type variables in the dataset
data.describe().T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1337.0</td>
      <td>39.222139</td>
      <td>14.044333</td>
      <td>18.0000</td>
      <td>27.000</td>
      <td>39.0000</td>
      <td>51.00000</td>
      <td>64.00000</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>1337.0</td>
      <td>30.663452</td>
      <td>6.100468</td>
      <td>15.9600</td>
      <td>26.290</td>
      <td>30.4000</td>
      <td>34.70000</td>
      <td>53.13000</td>
    </tr>
    <tr>
      <th>children</th>
      <td>1337.0</td>
      <td>1.095737</td>
      <td>1.205571</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>2.00000</td>
      <td>5.00000</td>
    </tr>
    <tr>
      <th>charges</th>
      <td>1337.0</td>
      <td>13279.121487</td>
      <td>12110.359656</td>
      <td>1121.8739</td>
      <td>4746.344</td>
      <td>9386.1613</td>
      <td>16657.71745</td>
      <td>63770.42801</td>
    </tr>
  </tbody>
</table>
</div>



**Categorical type Summary**


```python
data.describe(include=['category']).T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sex</th>
      <td>1337</td>
      <td>2</td>
      <td>male</td>
      <td>675</td>
    </tr>
    <tr>
      <th>smoker</th>
      <td>1337</td>
      <td>2</td>
      <td>no</td>
      <td>1063</td>
    </tr>
    <tr>
      <th>region</th>
      <td>1337</td>
      <td>4</td>
      <td>southeast</td>
      <td>364</td>
    </tr>
  </tbody>
</table>
</div>



**Observations**
* `sex` has **2** unique categories.  
    * `male` has the highest frequency. 
    
    
* `smoker` has **2** unique categories.  
    * `no` has the highest frequency.  
    
    
* `region` has **4** unique categories.  
    * `southeast` has the highest frequency.

---

**Create independent sub-lists to separate Numerical and Categorical variables for EDA**


```python
# Select numeric variables
numeric_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()
# Select categorical variables
categorical_columns = data.select_dtypes(include=['category']).columns.tolist()
```

---

## Numerical data

**Skew Summary**


```python
# Display the skew summary for the numerical variables
for var in data[numeric_columns].columns:
    var_skew = data[var].skew()
    if var_skew > 1:
        print(f"The '{var}' distribution is highly right skewed.\n")
    elif var_skew < -1:
        print(f"The '{var}' distribution is highly left skewed.\n")
    elif (var_skew > 0.5) & (var_skew < 1):
        print(f"The '{var}' distribution is moderately right skewed.\n")
    elif (var_skew < -0.5) & (var_skew > -1):
        print(f"The '{var}' distribution is moderately left skewed.\n")
    else:
        print(f"The '{var}' distribution is fairly symmetrical.\n")
```

    The 'age' distribution is fairly symmetrical.
    
    The 'bmi' distribution is fairly symmetrical.
    
    The 'children' distribution is moderately right skewed.
    
    The 'charges' distribution is highly right skewed.
    
    

**Outlier check function**


```python
# Outlier check
def outlier_count(data):
    """
    This function checks the lower and upper 
    outliers for all numerical variables.

    Outliers are found where data points exists either:
    - Greater than `1.5*IQR` above the 75th percentile
    - Less than `1.5*IQR` below the 25th percentile
    """
    numeric = data.select_dtypes(include=np.number).columns.to_list()
    for i in numeric:
        # Get name of series
        name = data[i].name
        # Calculate the IQR for all values and omit NaNs
        IQR = spy.stats.iqr(data[i], nan_policy="omit")
        # Calculate the boxplot upper fence
        upper_fence = data[i].quantile(0.75) + 1.5 * IQR
        # Calculate the boxplot lower fence
        lower_fence = data[i].quantile(0.25) - 1.5 * IQR
        # Calculate the count of outliers above upper fence
        upper_outliers = data[i][data[i] > upper_fence].count()
        # Calculate the count of outliers below lower fence
        lower_outliers = data[i][data[i] < lower_fence].count()
        # Check if there are no outliers
        if (upper_outliers == 0) & (lower_outliers == 0):
            continue
        print(
            f"The '{name}' distribution has '{lower_outliers}' lower outliers and '{upper_outliers}' upper outliers.\n"
        )
```


```python
#Applying the Outlier check function for the sub-dataframe of numerical variables
outlier_count(data[numeric_columns])
```

    The 'bmi' distribution has '0' lower outliers and '9' upper outliers.
    
    The 'charges' distribution has '0' lower outliers and '139' upper outliers.
    
    

---

## Categorical data

**Unique states**


```python
# Display the unique values for all categorical variables
for i in categorical_columns:
    print('Unique values in', i, 'are :')
    print(data[i].value_counts())
    print('--'*55)
```

    Unique values in sex are :
    sex
    male      675
    female    662
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in smoker are :
    smoker
    no     1063
    yes     274
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in region are :
    region
    southeast    364
    southwest    325
    northeast    324
    northwest    324
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    

---

# Heat Map

Create sub-dataframe removing binned numerical variables


```python
onehot = pd.get_dummies(data,
                        columns=data.select_dtypes(include=['category']).columns.tolist())
oh_corr = onehot.corr()

annot_kws = {"fontsize": 12}

symmetric_matrix = (oh_corr + oh_corr.T) / 2

# Create a mask for the upper half of the matrix
mask = np.triu(np.ones_like(symmetric_matrix), k=1)

plt.figure(figsize=(16, 12))
sns.heatmap(oh_corr, annot=True, fmt=".2f", mask=mask,
            cmap='coolwarm', square=True, annot_kws=annot_kws)
plt.yticks(rotation=0)
plt.show()
```


    
![png](output_59_0.png)
    


**Observations:**
* `Age` has a positive correlation with `Charges` which indicates that as the customers get older they file more medical claims.
* `BMI` has positive but weak correlations with `Charges` and `SouthEast region`.
* `Non-smokers` file lower medical claims than `Smokers`. This is observed by the highly positive correlation between `Charges` and `Smokers` while there is also a highly negative correlation between `Charges` and `Non-smokers`.

---

<span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:black;">Charges vs Age

**Relationship between Charges and Age with Smoker**


```python
sns.lmplot(x="age", y="charges", hue="smoker", data=data, markers=["*", "+"])
plt.title("Charges versus Age by Smoker")
plt.show()
```


    
![png](output_64_0.png)
    


**Observations:**
* Customers who are `smokers` file higher medical insurance claims than `non-smokers`.

<span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:black;">Charges vs BMI

**Relationship between Charges and BMI with Smoker**


```python
sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, markers=["*", "+"]);
plt.title("Charges versus BMI by Smoker")
plt.show();
```


    
![png](output_68_0.png)
    


**Observations:**
* As a customer's BMI increases, the magnitude of the medical claims filed significantly increases if they are a `smoker` as well.
* For `non-smokers`, as the customer's BMI increases there isn't any significant change at all in the size of the medical claims filed.


---

# Feature Engineering

Binning Continuous data into Categorical Data

**Categorizing Age**

Let's categorize the **AGE** into groups for analysis
Let's split age into the following groups:
1. Young Adults (ages 18-34)
2. Middle Age (ages 35-54)
3. Senior Adults (ages 55-64)


```python
# Define the age bins and labels
age_bins = [18, 35, 55, 64]
age_label = ["Young Adults", "Middle Aged", "Senior Adults"]
# Create a new column with age categories
data['AgeGroup'] = pd.cut(
    data['age'], bins=age_bins, include_lowest=True, right=False, labels=age_label)
```

**Categorizing BMI**

Let's categorize the **BMI** into groups for analysis
Let's split bmi into the following groups:
1. Under Weight (bmi < 18.5)
2. Ideal BMI (bmi 18.5-24.9)
3. Over Weight (bmi 25-29.9)
4. Obese (bmi > 30)


```python
# Define the age bins and labels
bmi_bins = [0, 18.5, 25, 30, 99]
bmi_label = ["Under Weight", "Ideal BMI", "Over Weight", "Obese"]
# Create a new column with age categories
data['bmi_groups'] = pd.cut(
    data.bmi, bmi_bins, include_lowest=True, right=False, labels=bmi_label)
```

---

Create sub-dataframe removing binned numerical variables


```python
data_binned = data.drop(columns=["age", 'bmi'])
```


```python
onehot = pd.get_dummies(data_binned,
                        columns=data_binned.select_dtypes(include=['category']).columns.tolist())
oh_corr = onehot.corr()

annot_kws = {"fontsize": 12}

symmetric_matrix = (oh_corr + oh_corr.T) / 2

# Create a mask for the upper half of the matrix
mask = np.triu(np.ones_like(symmetric_matrix), k=1)

plt.figure(figsize=(16, 12))
sns.heatmap(oh_corr, annot=True, fmt=".2f", mask=mask,
            cmap='coolwarm', square=True, annot_kws=annot_kws)
plt.yticks(rotation=0)
plt.show()
```


    
![png](output_82_0.png)
    


---

# Exploratory Data Analysis

## Univariate Analysis

### Numerical Variables

**Histogram Overview**


```python
def histogram_overview(data):
    """
    Histogram Overview function
    
    This function below generates a subplots of  `histogram plots` & 
    showing the `distribution of the numerical varible input`
    
    * Generates subplots for each numerical variable in a three column structure.
    * The function takes the Pandas dataframe as the input
    * The function selects the numerical variables from the applied dataframe.
    * It generates a vertical `line` to indicate the `mean`, `median` and `mode` on the histogram
    * `sns.set_style` - sets the Seaborn theme
    * `subplot_nrows` - calculated number of subplot rows
    * `subplot_ncols` - configures the three column structure subplots
    * `figsize` - indicates the size of the plot
    * `sns.reset_defaults()` - resets Seaborn themes and settings to default
    
    """
    num_vars = data.select_dtypes(include=np.number).columns.to_list()
    plt.figure(figsize=(10, 10))
    for i in range(len(num_vars)):
        # Set seaborn theme
        sns.set_style("darkgrid")
        # Subplot no of columns
        subplot_ncols = math.ceil(np.sqrt(len(num_vars)))
        # Subplot no of rows
        subplot_nrows = subplot_ncols

        plt.subplot(subplot_nrows, subplot_ncols, i + 1)

        plt.hist(data[num_vars[i]])
        #Plot vertical line for the mean
        plt.axvline(data[num_vars[i]].mean(),
                    color='green',
                    linestyle='--',
                    label="mean")
        #Plot vertical line for the median
        plt.axvline(data[num_vars[i]].median(),
                    color='red',
                    linestyle='-',
                    label="median")
        #Plot vertical line for the mode
        plt.axvline(data[num_vars[i]].mode()[0],
                    color='black',
                    linestyle='-',
                    label="mode")
        plt.legend()
        plt.tight_layout()
        plt.title(num_vars[i], fontsize=16)
    plt.show()
    # Reset seaborn theme
    sns.reset_defaults()
```


```python
histogram_overview(data)
```


    
![png](output_89_0.png)
    


**Boxplot overview**


```python
def boxplot_overview(data):
    """
    This function below generates a subplots of `box plots` &
    showing the `distribution of the numerical variable input with outliers`.

    * Generates subplots for each numerical variable in a three column structure.
    * The function takes the Pandas dataframe as the input
    * The function selects the numerical variables from the applied dataframe.
    * It shows the `mean` in the boxplot.
    * `sns.set_style` - sets the Seaborn theme
    * `subplot_nrows` - calculated number of subplot rows
    * `subplot_ncols` - configures the three column structure subplots
    * `figsize` - indicates the size of the plot
    * `sns.reset_defaults()` - resets Seaborn themes and settings to default
    """

    num_vars = data.select_dtypes(include=np.number).columns.to_list()
    plt.figure(figsize=(10, 10))
    for i in range(len(num_vars)):
        # Set seaborn theme
        sns.set_style("darkgrid")
        # Subplot no of columns
        subplot_ncols = math.ceil(np.sqrt(len(num_vars)) )  
        # Subplot no of rows
        subplot_nrows = math.ceil(len(num_vars) / subplot_ncols)  
          
        plt.subplot(subplot_nrows, subplot_ncols, i + 1)
        sns.boxplot(y=data[num_vars[i]], width=0.3, showmeans=True)
        plt.tight_layout()
        plt.title(num_vars[i], fontsize=16)
    plt.show()
    # Reset seaborn theme
    sns.reset_defaults()
```


```python
boxplot_overview(data)
```


    
![png](output_92_0.png)
    


### Categorical Variables


```python
def bar_chart(data):
    """
    This function below generates a `bar chart` showing
    the `distribution of the categorical varible input`.
    * The function also `generates an image file` of the plot.
    * The function takes the Pandas series as the input.
    * It `computes the frequency of each unique element` and 
      displays the distribution of the elements to in horizontal bars.
    * The `percentage of each bar` is also calculated and placed to 
      the right end of each bar.
    * `sns.despine()` - removes the upper and right border of the chart
    * For each horizontal bar the width is calculated as a percentage of
      the entire quanta of datapoints.
    * The percentage is annotated to the each bar by plotting the cardinal locations.

    """

    # Create a horizontal count plot while sorting variables in descending order
    g = sns.countplot(y=data, order = data.astype(str).value_counts().index)
    # Remove the top and right spines from plot
    sns.despine()
    # length of the column
    col_length = len(data)
    for p in g.patches:
        # percentage of each class of the category
        percentage = '{:.1f}%'.format(100 * p.get_width()/col_length)
        # width of the plot
        x = p.get_x() + p.get_width() + 0.02
        # height of the plot
        y = p.get_y() + p.get_height()/2
        # annotate the percentage
        g.annotate(percentage, (x, y), size=12)
        plt.title("Distribution of {}".format(
            data.name), loc="center", fontsize=22)
    plt.show()
    # Line separator
    print('--'*55)
```


```python
data.select_dtypes(exclude = np.number).columns.to_list()
```




    ['sex', 'smoker', 'region', 'AgeGroup', 'bmi_groups']




```python
for each_var in data.select_dtypes(exclude = np.number).columns.to_list():
    plt.figure()
    bar_chart(data=data[each_var])
```


    
![png](output_96_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_96_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_96_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_96_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_96_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    

---

---
