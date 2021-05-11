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

## Exploratory Data Analysis


### Check dataset unique values
``` python
# Train
counts_train = train_raw.nunique()
counts_train
```

```python
# Test
counts_test = test_raw.nunique()
counts_test
```

$\bullet$ As we can see the train dataset & test dataset are in the same structure and both contain nested columns 

$\bullet$ we will unpack the nested structures later on. Those columns are all Strings.

$\bullet$ There are no duplicate rows in both datasets

$\bullet$ All movies are 'Realesed' and therfore <b>Status</b> column, containing only 1 value, has no impact and won't be in use
