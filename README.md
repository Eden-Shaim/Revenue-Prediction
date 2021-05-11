# Revenue-Prediction
HW1 - Lab 2 course in data science

``` python

import numpy as np
import pandas as pd
from preprocess import *
pd.set_option('display.max_columns', None)
from matplotlib.colors import rgb_to_hsv
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(font_scale=1.3)
```

``` python
folder_path_data = 'hw1_data/'
train_raw = pd.read_csv(folder_path_data +'train.tsv',delimiter='\t')
test_raw = pd.read_csv(folder_path_data + 'test.tsv',delimiter='\t')
```
