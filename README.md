# TASK #1: INTRO


# TASK #2: IMPORT DATASETS AND LIBRARIES


```python
# The World Happiness Report determines the state of global happiness. 
# The happiness scores and rankings data has been collected by asking individuals to rank their life.
# Ranking ranges from 0 (worst possible life) to 10 (best possible life). 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False)
```


```python
# Import csv file into pandas dataframe
happy_df = pd.read_csv('happiness_report.csv')
```


```python
# print the first 5 rows of the dataframe
happy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>



**PRACTICE OPPORTUNITY #1 [OPTIONAL]:** 
- **Select 2 countries from the dataframe and explore scores. Perform sanity check.**


```python

```

# TASK #3: PERFORM EXPLORATORY DATA ANALYSIS


```python
# Check the number of non-null values in the dataframe
happy_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 156 entries, 0 to 155
    Data columns (total 9 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   Overall rank                  156 non-null    int64  
     1   Country or region             156 non-null    object 
     2   Score                         156 non-null    float64
     3   GDP per capita                156 non-null    float64
     4   Social support                156 non-null    float64
     5   Healthy life expectancy       156 non-null    float64
     6   Freedom to make life choices  156 non-null    float64
     7   Generosity                    156 non-null    float64
     8   Perceptions of corruption     156 non-null    float64
    dtypes: float64(7), int64(1), object(1)
    memory usage: 11.1+ KB
    


```python
# Check Null values
happy_df.isnull().sum()
```




    Overall rank                    0
    Country or region               0
    Score                           0
    GDP per capita                  0
    Social support                  0
    Healthy life expectancy         0
    Freedom to make life choices    0
    Generosity                      0
    Perceptions of corruption       0
    dtype: int64




```python
# Obtain the Statistical summary of the dataframe
happy_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.500000</td>
      <td>5.407096</td>
      <td>0.905147</td>
      <td>1.208814</td>
      <td>0.725244</td>
      <td>0.392571</td>
      <td>0.184846</td>
      <td>0.110603</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.177428</td>
      <td>1.113120</td>
      <td>0.398389</td>
      <td>0.299191</td>
      <td>0.242124</td>
      <td>0.143289</td>
      <td>0.095254</td>
      <td>0.094538</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.853000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.750000</td>
      <td>4.544500</td>
      <td>0.602750</td>
      <td>1.055750</td>
      <td>0.547750</td>
      <td>0.308000</td>
      <td>0.108750</td>
      <td>0.047000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>78.500000</td>
      <td>5.379500</td>
      <td>0.960000</td>
      <td>1.271500</td>
      <td>0.789000</td>
      <td>0.417000</td>
      <td>0.177500</td>
      <td>0.085500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>117.250000</td>
      <td>6.184500</td>
      <td>1.232500</td>
      <td>1.452500</td>
      <td>0.881750</td>
      <td>0.507250</td>
      <td>0.248250</td>
      <td>0.141250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>156.000000</td>
      <td>7.769000</td>
      <td>1.684000</td>
      <td>1.624000</td>
      <td>1.141000</td>
      <td>0.631000</td>
      <td>0.566000</td>
      <td>0.453000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the number of duplicated entries in the dataframe
happy_df.duplicated().sum() # since there are no duplicates, no further action is required
```




    0



**PRACTICE OPPORTUNITY #2:** 
- **Which country has the maximum happiness score? What is the perception of corruption in this country?**


```python

```

# TASK #4: PERFORM DATA VISUALIZATION: PAIRPLOT & SCATTERMATRIX


```python
# A scatterplot matrix is a matrix associated to n numerical arrays (data variables), $X_1,X_2,…,X_n$ , of the same length. 
# The cell (i,j) of such a matrix displays the scatter plot of the variable Xi versus Xj.
# Here we show the Plotly Express function px.scatter_matrix to plot the scatter matrix for the columns of the dataframe. By default, all columns are considered.

# Note:
# Positive correlation between GDP and score 
# Positive correlation between Social Support and score 


fig = px.scatter_matrix(happy_df[['Score','GDP per capita', 'Social support', 'Healthy life expectancy', 
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']], width = 1500, height = 1500)
fig.show()
```

![png](np01.png)


```python
# Alternatively, you can use Seaborn to plot the pairplots as follows (Note that the plot is no longer interactive): 
fig = plt.figure(figsize = (20,20))
sns.pairplot(happy_df[['Score','GDP per capita', 'Social support', 'Healthy life expectancy', 
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']])

# Positive correlation between GDP and score 
# Positive correlation between Social Support and score 

```




    <seaborn.axisgrid.PairGrid at 0x17ab7ff2c88>




    <Figure size 1440x1440 with 0 Axes>



![png](output_18_2.png)


# TASK #5: PERFORM DATA VISUALIZATION: DISTPLOT & CORRELATION MATRIX


```python
# distplot combines the matplotlib.hist function with seaborn kdeplot()

columns = ['Score','GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']

plt.figure(figsize = (20, 50))
for i in range(len(columns)):
  plt.subplot(8, 2, i+1)
  sns.distplot(happy_df[columns[i]], color = 'g');
  plt.title(columns[i])

plt.tight_layout()
```


![png](output_20_0.png)



```python
fig = px.imshow(happy_df.corr())
fig.show()
```

![png](np02.png)



```python
# Get the correlation matrix
corr_matrix = happy_df.corr()
corr_matrix
sns.heatmap(corr_matrix, annot = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17abcfb8160>




![png](output_22_1.png)


# TASK #6: PERFORM DATA VISUALIZATION: SCATTERPLOTS AND BUBBLE CHARTS


```python
# Plot the relationship between score, GDP and region
fig = px.scatter(happy_df, x = 'GDP per capita', y = 'Score', text = 'Country or region')
fig.update_traces(textposition = 'top center')
fig.update_layout(height = 1000)
fig.show()
```

![png](np03.png)



```python
# Plot the relationship between score and GDP (while adding color and size)
fig = px.scatter(happy_df, x = "GDP per capita", y = "Score", size = 'Overall rank', color = "Country or region", hover_name = "Country or region")

fig.update_layout(title_text = 'Happiness Score vs GDP per Capita')
fig.show()
```

![png](np04.png)




```python
# Plot the relationship between score and freedom to make life choices

fig = px.scatter(happy_df, x = 'Freedom to make life choices', y = "Score", size = 'Overall rank', color = "Country or region", hover_name = "Country or region",
          trendline = "ols")

fig.update_layout(title_text = 'Happiness Score vs Freedom to make life choices')
fig.show()
```

![png](np05.png)


# FINAL CAPSTONE PROJECT 

Using "cars.csv" dataset included in the guided project package, please complete the following tasks: 
- 1. Using Pandas, read the "cars.csv" dataset
- 2. Perform exploratory data analysis
- 3. Remove $ sign and comma (,) from MSRP and Invoice columns
- 4. Convert MSRP and Invoice columns to integer datatypes and perform sanity check on the data
- 5. Plot the scattermatrix and pairplot
- 6. Plot a scatterplot between 'Horsepower' and 'MSRP' while showing 'Make' as text. Use the 'Cylinders' column to display color.
- 7. Plot the wordcloud of the Make column
- 8. Plot the histogram of Make and Type of the car using Plotly Express
- 9. Find out which manufacturer has high number of Sports type 
- 10. Find out which manufacturers has Hybrids
- 11. Plot the correlation matrix using plotly express and Seaborn
- 12. Comment on the correlation matrix, which feature has the highest positive correlation with MSRP?

# PRACTICE OPPORTUNITIES SOLUTIONS

**PRACTICE OPPORTUNITY #1 SOLUTION:**
- **Select 2 countries from the dataframe and explore scores. Perform sanity check.**


```python
happy_df[happy_df['Country or region']=='Canada']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Canada</td>
      <td>7.278</td>
      <td>1.365</td>
      <td>1.505</td>
      <td>1.039</td>
      <td>0.584</td>
      <td>0.285</td>
      <td>0.308</td>
    </tr>
  </tbody>
</table>
</div>




```python
happy_df[happy_df['Country or region']=='Zimbabwe']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145</th>
      <td>146</td>
      <td>Zimbabwe</td>
      <td>3.663</td>
      <td>0.366</td>
      <td>1.114</td>
      <td>0.433</td>
      <td>0.361</td>
      <td>0.151</td>
      <td>0.089</td>
    </tr>
  </tbody>
</table>
</div>



**PRACTICE OPPORTUNITY #2 SOLUTION:**
- **Which country has the maximum happiness score? What is the perception of corruption in this country?**


```python
happy_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.500000</td>
      <td>5.407096</td>
      <td>0.905147</td>
      <td>1.208814</td>
      <td>0.725244</td>
      <td>0.392571</td>
      <td>0.184846</td>
      <td>0.110603</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.177428</td>
      <td>1.113120</td>
      <td>0.398389</td>
      <td>0.299191</td>
      <td>0.242124</td>
      <td>0.143289</td>
      <td>0.095254</td>
      <td>0.094538</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.853000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.750000</td>
      <td>4.544500</td>
      <td>0.602750</td>
      <td>1.055750</td>
      <td>0.547750</td>
      <td>0.308000</td>
      <td>0.108750</td>
      <td>0.047000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>78.500000</td>
      <td>5.379500</td>
      <td>0.960000</td>
      <td>1.271500</td>
      <td>0.789000</td>
      <td>0.417000</td>
      <td>0.177500</td>
      <td>0.085500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>117.250000</td>
      <td>6.184500</td>
      <td>1.232500</td>
      <td>1.452500</td>
      <td>0.881750</td>
      <td>0.507250</td>
      <td>0.248250</td>
      <td>0.141250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>156.000000</td>
      <td>7.769000</td>
      <td>1.684000</td>
      <td>1.624000</td>
      <td>1.141000</td>
      <td>0.631000</td>
      <td>0.566000</td>
      <td>0.453000</td>
    </tr>
  </tbody>
</table>
</div>




```python
happy_df[happy_df['Score'] == 7.769000]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.34</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# FINAL CAPSTONE PROJECT SOLUTION


```python
import numpy as np # Multi-dimensional array object
import pandas as pd # Data Manipulation
import seaborn as sns # Data Visualization
import matplotlib.pyplot as plt # Data Visualization
import plotly.express as px # Interactive Data Visualization

```


```python
# Read the CSV file 
car_df = pd.read_csv("cars.csv")
```


```python
# Load the top 10 instances
car_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Type</th>
      <th>Origin</th>
      <th>DriveTrain</th>
      <th>MSRP</th>
      <th>Invoice</th>
      <th>EngineSize</th>
      <th>Cylinders</th>
      <th>Horsepower</th>
      <th>MPG_City</th>
      <th>MPG_Highway</th>
      <th>Weight</th>
      <th>Wheelbase</th>
      <th>Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>MDX</td>
      <td>SUV</td>
      <td>Asia</td>
      <td>All</td>
      <td>$36,945</td>
      <td>$33,337</td>
      <td>3.5</td>
      <td>6.0</td>
      <td>265</td>
      <td>17</td>
      <td>23</td>
      <td>4451</td>
      <td>106</td>
      <td>189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Acura</td>
      <td>RSX Type S 2dr</td>
      <td>Sedan</td>
      <td>Asia</td>
      <td>Front</td>
      <td>$23,820</td>
      <td>$21,761</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>200</td>
      <td>24</td>
      <td>31</td>
      <td>2778</td>
      <td>101</td>
      <td>172</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Acura</td>
      <td>TSX 4dr</td>
      <td>Sedan</td>
      <td>Asia</td>
      <td>Front</td>
      <td>$26,990</td>
      <td>$24,647</td>
      <td>2.4</td>
      <td>4.0</td>
      <td>200</td>
      <td>22</td>
      <td>29</td>
      <td>3230</td>
      <td>105</td>
      <td>183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Acura</td>
      <td>TL 4dr</td>
      <td>Sedan</td>
      <td>Asia</td>
      <td>Front</td>
      <td>$33,195</td>
      <td>$30,299</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>270</td>
      <td>20</td>
      <td>28</td>
      <td>3575</td>
      <td>108</td>
      <td>186</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Acura</td>
      <td>3.5 RL 4dr</td>
      <td>Sedan</td>
      <td>Asia</td>
      <td>Front</td>
      <td>$43,755</td>
      <td>$39,014</td>
      <td>3.5</td>
      <td>6.0</td>
      <td>225</td>
      <td>18</td>
      <td>24</td>
      <td>3880</td>
      <td>115</td>
      <td>197</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Acura</td>
      <td>3.5 RL w/Navigation 4dr</td>
      <td>Sedan</td>
      <td>Asia</td>
      <td>Front</td>
      <td>$46,100</td>
      <td>$41,100</td>
      <td>3.5</td>
      <td>6.0</td>
      <td>225</td>
      <td>18</td>
      <td>24</td>
      <td>3893</td>
      <td>115</td>
      <td>197</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Acura</td>
      <td>NSX coupe 2dr manual S</td>
      <td>Sports</td>
      <td>Asia</td>
      <td>Rear</td>
      <td>$89,765</td>
      <td>$79,978</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>290</td>
      <td>17</td>
      <td>24</td>
      <td>3153</td>
      <td>100</td>
      <td>174</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Audi</td>
      <td>A4 1.8T 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$25,940</td>
      <td>$23,508</td>
      <td>1.8</td>
      <td>4.0</td>
      <td>170</td>
      <td>22</td>
      <td>31</td>
      <td>3252</td>
      <td>104</td>
      <td>179</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Audi</td>
      <td>A41.8T convertible 2dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$35,940</td>
      <td>$32,506</td>
      <td>1.8</td>
      <td>4.0</td>
      <td>170</td>
      <td>23</td>
      <td>30</td>
      <td>3638</td>
      <td>105</td>
      <td>180</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Audi</td>
      <td>A4 3.0 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$31,840</td>
      <td>$28,846</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>220</td>
      <td>20</td>
      <td>28</td>
      <td>3462</td>
      <td>104</td>
      <td>179</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Load the bottom 10 instances 
car_df.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Type</th>
      <th>Origin</th>
      <th>DriveTrain</th>
      <th>MSRP</th>
      <th>Invoice</th>
      <th>EngineSize</th>
      <th>Cylinders</th>
      <th>Horsepower</th>
      <th>MPG_City</th>
      <th>MPG_Highway</th>
      <th>Weight</th>
      <th>Wheelbase</th>
      <th>Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>418</th>
      <td>Volvo</td>
      <td>S60 2.5 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>All</td>
      <td>$31,745</td>
      <td>$29,916</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>208</td>
      <td>20</td>
      <td>27</td>
      <td>3903</td>
      <td>107</td>
      <td>180</td>
    </tr>
    <tr>
      <th>419</th>
      <td>Volvo</td>
      <td>S60 T5 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$34,845</td>
      <td>$32,902</td>
      <td>2.3</td>
      <td>5.0</td>
      <td>247</td>
      <td>20</td>
      <td>28</td>
      <td>3766</td>
      <td>107</td>
      <td>180</td>
    </tr>
    <tr>
      <th>420</th>
      <td>Volvo</td>
      <td>S60 R 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>All</td>
      <td>$37,560</td>
      <td>$35,382</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>300</td>
      <td>18</td>
      <td>25</td>
      <td>3571</td>
      <td>107</td>
      <td>181</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Volvo</td>
      <td>S80 2.9 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$37,730</td>
      <td>$35,542</td>
      <td>2.9</td>
      <td>6.0</td>
      <td>208</td>
      <td>20</td>
      <td>28</td>
      <td>3576</td>
      <td>110</td>
      <td>190</td>
    </tr>
    <tr>
      <th>422</th>
      <td>Volvo</td>
      <td>S80 2.5T 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>All</td>
      <td>$37,885</td>
      <td>$35,688</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>194</td>
      <td>20</td>
      <td>27</td>
      <td>3691</td>
      <td>110</td>
      <td>190</td>
    </tr>
    <tr>
      <th>423</th>
      <td>Volvo</td>
      <td>C70 LPT convertible 2dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$40,565</td>
      <td>$38,203</td>
      <td>2.4</td>
      <td>5.0</td>
      <td>197</td>
      <td>21</td>
      <td>28</td>
      <td>3450</td>
      <td>105</td>
      <td>186</td>
    </tr>
    <tr>
      <th>424</th>
      <td>Volvo</td>
      <td>C70 HPT convertible 2dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$42,565</td>
      <td>$40,083</td>
      <td>2.3</td>
      <td>5.0</td>
      <td>242</td>
      <td>20</td>
      <td>26</td>
      <td>3450</td>
      <td>105</td>
      <td>186</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Volvo</td>
      <td>S80 T6 4dr</td>
      <td>Sedan</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$45,210</td>
      <td>$42,573</td>
      <td>2.9</td>
      <td>6.0</td>
      <td>268</td>
      <td>19</td>
      <td>26</td>
      <td>3653</td>
      <td>110</td>
      <td>190</td>
    </tr>
    <tr>
      <th>426</th>
      <td>Volvo</td>
      <td>V40</td>
      <td>Wagon</td>
      <td>Europe</td>
      <td>Front</td>
      <td>$26,135</td>
      <td>$24,641</td>
      <td>1.9</td>
      <td>4.0</td>
      <td>170</td>
      <td>22</td>
      <td>29</td>
      <td>2822</td>
      <td>101</td>
      <td>180</td>
    </tr>
    <tr>
      <th>427</th>
      <td>Volvo</td>
      <td>XC70</td>
      <td>Wagon</td>
      <td>Europe</td>
      <td>All</td>
      <td>$35,145</td>
      <td>$33,112</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>208</td>
      <td>20</td>
      <td>27</td>
      <td>3823</td>
      <td>109</td>
      <td>186</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display the feature columns
car_df.columns
```




    Index(['Make', 'Model', 'Type', 'Origin', 'DriveTrain', 'MSRP', 'Invoice',
           'EngineSize', 'Cylinders', 'Horsepower', 'MPG_City', 'MPG_Highway',
           'Weight', 'Wheelbase', 'Length'],
          dtype='object')




```python
# Check if any missing values are present in the dataframe
car_df.isnull().sum()

```




    Make           0
    Model          0
    Type           0
    Origin         0
    DriveTrain     0
    MSRP           0
    Invoice        0
    EngineSize     0
    Cylinders      2
    Horsepower     0
    MPG_City       0
    MPG_Highway    0
    Weight         0
    Wheelbase      0
    Length         0
    dtype: int64




```python
# Obtain the summary of the dataframe
car_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 428 entries, 0 to 427
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Make         428 non-null    object 
     1   Model        428 non-null    object 
     2   Type         428 non-null    object 
     3   Origin       428 non-null    object 
     4   DriveTrain   428 non-null    object 
     5   MSRP         428 non-null    object 
     6   Invoice      428 non-null    object 
     7   EngineSize   428 non-null    float64
     8   Cylinders    426 non-null    float64
     9   Horsepower   428 non-null    int64  
     10  MPG_City     428 non-null    int64  
     11  MPG_Highway  428 non-null    int64  
     12  Weight       428 non-null    int64  
     13  Wheelbase    428 non-null    int64  
     14  Length       428 non-null    int64  
    dtypes: float64(2), int64(6), object(7)
    memory usage: 50.3+ KB
    


```python
# Convert MSRP and Invoice datatype to integer so we need to remove $ sign and comma (,) from these 2 columns

car_df["MSRP"] = car_df["MSRP"].str.replace("$", "")
car_df["MSRP"] = car_df["MSRP"].str.replace(",", "")
car_df["MSRP"] = car_df["MSRP"].astype(int)

car_df["MSRP"]

car_df["Invoice"] = car_df["Invoice"].str.replace("$", "")
car_df["Invoice"] = car_df["Invoice"].str.replace(",", "")
car_df["Invoice"] = car_df["Invoice"].astype(int)

# Let's view the updated MSRP and Invoice Columns
car_df.head()

# Display the updated summary of the dataframe
car_df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 428 entries, 0 to 427
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Make         428 non-null    object 
     1   Model        428 non-null    object 
     2   Type         428 non-null    object 
     3   Origin       428 non-null    object 
     4   DriveTrain   428 non-null    object 
     5   MSRP         428 non-null    int32  
     6   Invoice      428 non-null    int32  
     7   EngineSize   428 non-null    float64
     8   Cylinders    426 non-null    float64
     9   Horsepower   428 non-null    int64  
     10  MPG_City     428 non-null    int64  
     11  MPG_Highway  428 non-null    int64  
     12  Weight       428 non-null    int64  
     13  Wheelbase    428 non-null    int64  
     14  Length       428 non-null    int64  
    dtypes: float64(2), int32(2), int64(6), object(5)
    memory usage: 46.9+ KB
    


```python
fig = px.scatter_matrix(car_df, width = 2000, height = 2000)
fig.show()

```


![png](np06.png)




```python
# Alternatively, you can use scatterplots to show the joint relationships and histograms for univariate distributions
sns.pairplot(data = car_df) 
```




    <seaborn.axisgrid.PairGrid at 0x17abd6c8f28>




![png](output_47_1.png)



```python

fig = px.scatter(car_df, x = 'Horsepower', y = 'MSRP', text = car_df['Make'], color = car_df['Cylinders'])
fig.update_traces(textposition = 'top center')
fig.update_layout(height = 2000)
fig.update_layout(width = 2000)

fig.show()
```


![png](np07.png)




```python
# Let's view various makes of the cars
car_df.Make.unique()

```




    array(['Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet',
           'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Hummer', 'Hyundai',
           'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover',
           'Lexus', 'Lincoln', 'MINI', 'Mazda', 'Mercedes-Benz', 'Mercury',
           'Mitsubishi', 'Nissan', 'Oldsmobile', 'Pontiac', 'Porsche', 'Saab',
           'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen',
           'Volvo'], dtype=object)




```python
fig = px.histogram(car_df, x = "Make",
                  labels = {"Make":"Manufacturer"},
                  title = "MAKE OF THE CAR",
                  color_discrete_sequence = ["maroon"])
fig.show()

```


![png](np08.png)




```python
# Let's view various types of the cars
car_df.Type.unique()

fig = px.histogram(car_df, x = "Type",
                  labels = {"Type":"Type"},
                  title = "TYPE OF THE CAR",
                  color_discrete_sequence = ["blue"])
                  
fig.show()
```

![png](np09.png)



```python
# Let's plot the location
car_df.Origin.unique()

fig = px.histogram(car_df, x = "Origin",
                  labels = {"Origin":"Origin"},
                  title = "LOCATION OF THE CAR SALES",
                  color_discrete_sequence = ["brown"])
                  
fig.show()

```



![png](np10.png)




```python
# Let's view the drivetrain of the cars
car_df.DriveTrain.unique()

fig = px.histogram(car_df, x = "DriveTrain",
                  labels = {"DriveTrain":"Drivetrain"},
                  title = "DRIVETRAIN OF THE CAR",
                  color_discrete_sequence = ["BLACK"])
                  
fig.show()
```

![png](np11.png)



```python
# Plot the make of the car and its location
fig = px.histogram(car_df, x = "Make",
                  color = "Origin",
                  labels = {"Make":"Manufacturer"},
                  title = "MAKE OF THE CAR Vs LOCATION")
                  
fig.show()
```


![png](np12.png)



```python
# Let's view the model of all used cars using WordCloud generator
from wordcloud import WordCloud, STOPWORDS

car_df

text = car_df.Model.values

stopwords = set(STOPWORDS)

wc = WordCloud(background_color = "black", max_words = 2000, max_font_size = 100, random_state = 3, 
              stopwords = stopwords, contour_width = 3).generate(str(text))       

fig = plt.figure(figsize = (25, 15))
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
plt.show()

```


![png](output_55_0.png)



```python

# Obtain the correlation matrix
car_df.corr()

fig = px.imshow(car_df.corr())
fig.show()

```

![png](np13.png)



```python
plt.figure(figsize = (8,8))
sns.heatmap(car_df.corr(), cmap="YlGnBu", annot = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17ac1c0f160>




![png](output_57_1.png)



```python
fig = px.histogram(car_df, x = "Make",
                  color = "Type",
                  labels = {"Make":"Manufacturer"},
                  title = "MAKE AND TYPE OF THE CAR",
                  opacity = 1)
                  
fig.show()

# Porsche
# Honda and Toyota


# Positive correlation between engine size and number of cylinders
# Positive correlation between horsepower and number of cylinders
# highest positive correlation with MSRP is = horsepower

```


![png](np14.png)



# THE END!
