
<a href="https://colab.research.google.com/github/stevenkcolin/tensorflow/blob/master/2019_01_26_feature_sets.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
```


```
def preprocess_features(california_housing_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets
```


```
# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())
```

    Training examples summary:



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
      <th>latitude</th>
      <th>longitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>rooms_per_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.6</td>
      <td>-119.5</td>
      <td>28.5</td>
      <td>2638.5</td>
      <td>538.5</td>
      <td>1427.4</td>
      <td>500.1</td>
      <td>3.9</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.1</td>
      <td>2.0</td>
      <td>12.6</td>
      <td>2170.9</td>
      <td>421.0</td>
      <td>1165.2</td>
      <td>384.0</td>
      <td>1.9</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.5</td>
      <td>-124.3</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.9</td>
      <td>-121.8</td>
      <td>18.0</td>
      <td>1462.0</td>
      <td>296.0</td>
      <td>786.0</td>
      <td>281.0</td>
      <td>2.6</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.2</td>
      <td>-118.5</td>
      <td>28.0</td>
      <td>2130.0</td>
      <td>435.0</td>
      <td>1164.0</td>
      <td>410.0</td>
      <td>3.6</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.7</td>
      <td>-118.0</td>
      <td>37.0</td>
      <td>3141.2</td>
      <td>646.0</td>
      <td>1718.0</td>
      <td>602.0</td>
      <td>4.8</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>max</th>
      <td>42.0</td>
      <td>-114.3</td>
      <td>52.0</td>
      <td>37937.0</td>
      <td>6445.0</td>
      <td>35682.0</td>
      <td>6082.0</td>
      <td>15.0</td>
      <td>55.2</td>
    </tr>
  </tbody>
</table>
</div>


    Validation examples summary:



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
      <th>latitude</th>
      <th>longitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>rooms_per_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.7</td>
      <td>-119.6</td>
      <td>28.7</td>
      <td>2656.0</td>
      <td>541.6</td>
      <td>1434.8</td>
      <td>503.9</td>
      <td>3.9</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.1</td>
      <td>2.0</td>
      <td>12.6</td>
      <td>2201.6</td>
      <td>422.8</td>
      <td>1105.2</td>
      <td>385.9</td>
      <td>1.9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.5</td>
      <td>-124.3</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.9</td>
      <td>-121.8</td>
      <td>18.0</td>
      <td>1463.0</td>
      <td>299.0</td>
      <td>796.0</td>
      <td>283.0</td>
      <td>2.6</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.3</td>
      <td>-118.5</td>
      <td>29.0</td>
      <td>2121.5</td>
      <td>430.5</td>
      <td>1175.5</td>
      <td>408.0</td>
      <td>3.5</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.7</td>
      <td>-118.0</td>
      <td>37.0</td>
      <td>3168.5</td>
      <td>655.2</td>
      <td>1732.0</td>
      <td>610.0</td>
      <td>4.7</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41.9</td>
      <td>-114.6</td>
      <td>52.0</td>
      <td>32054.0</td>
      <td>5290.0</td>
      <td>15507.0</td>
      <td>5050.0</td>
      <td>15.0</td>
      <td>41.3</td>
    </tr>
  </tbody>
</table>
</div>


    Training targets summary:



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
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>207.8</td>
    </tr>
    <tr>
      <th>std</th>
      <td>115.8</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>120.1</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>181.1</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>265.8</td>
    </tr>
    <tr>
      <th>max</th>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>


    Validation targets summary:



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
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>206.2</td>
    </tr>
    <tr>
      <th>std</th>
      <td>116.4</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>118.8</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>178.4</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>262.9</td>
    </tr>
    <tr>
      <th>max</th>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>

