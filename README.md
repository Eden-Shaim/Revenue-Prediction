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
![svg](README_files/raw_train_head.jpeg)

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

- As we can see the train dataset & test dataset are in the same structure and both contain nested columns 
- we will unpack the nested structures later on. Those columns are all Strings.
- There are no duplicate rows in both datasets
- All movies are 'Realesed' and therfore <b>Status</b> column, containing only 1 value, has no impact and won't be in use

### Let's see some statistics about the numerical columns:
```python
train_raw.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))
```



```python
test_raw.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))
```


- We can see that both train & test contain uncertain data - rows with runtime = 0 , budget = 0 
- We will treat those values as missing data and they will be imputed later on.

```python
print(f"Train dataset: Amount of movies with 0 budget: {len(train_raw[train_raw['budget'] == 0])}")
print(f"Train dataset: Amount of movies with 0 runtime: {len(train_raw[train_raw['runtime'] == 0])}")
print(f"Test dataset : Amount of movies with 0 budget: {len(test_raw[test_raw['budget'] == 0])}")
print(f"Test dataset : Amount of movies with 0 budget: {len(test_raw[test_raw['runtime'] == 0])}")
```

### Features distribution & comparative analysis between features
``` python
sns.pairplot(train_raw.select_dtypes('number').drop(columns='id'), kind="reg", diag_kind="kde")
plt.show()
```


* In the main diagonal we can see each of the numerical column distribution

* From the pair-plot we can deduce info about the correlated columns :
- popullarity & vote_count
- revenue & budjet
- runtime & vote_count 
- revenue & popullarity
- ...

#### Let's dig a bit deeper about about correlations with Revenue (our target column)
``` python
cols =['revenue','budget','popularity','runtime']
sns.heatmap(train_raw[cols].corr(), cmap="Blues")
plt.show()
```


## Missing data
```python 
# The raw data contains Empty values as ['[]', '{}', '']. we will replace those values with None just for an easier view 
train_na = train_raw.replace(to_replace =['[]', '{}', ''], value = np.nan)
test_na = test_raw.replace(to_replace =['[]', '{}', ''], value = np.nan)
```

``` python
print ("Train missing values :")
train_na.isnull().sum().sort_values(ascending=False)
```

``` python
print ("Test missing values :")
test_na.isnull().sum().sort_values(ascending=False)
```
- The feature <b>'belongs_to_collection'</b> contains many None's. That makes sense because most the movies are not part of a collection 
- The features <b>'homepage'</b> & <b>'tagline'</b> also has many None values.
That makes them preaty unrelevant for out task because if imputation is required (not necessarily) we won't be able to apply it without additional data. Also the percentage of None values is ~50% & ~20% accordingly (high numbers)
- The features <b>'backdrop_path'</b> & <b>'poster_path'</b> refer to images which are irrelevant for our task 

## Feature Engineering

### Unpack nested columns
First, we should unpack all the nested attrubutes in the columns.

We used <b>eval</b> to convert the string representation of an attrubutes to an object.
Following, we extracted the relevant information (in our opinion) from each column as follows:

- `belongs_to_collection_ids` : If a movie belongs to a collection - we keep the collection id, else None.
- `genres` : The genre `name` attributes
- `production_comapnies_names` & `production_comapnies_origin_country` : List of production company names attribute & production companies origin country.
- `production_countries` : List of countries (`iso_3166`) where the movie was filmed.
- `release_month`, `release_year` : The month & year the film was released on.
- `spoken_languages_len` : Number of spoken languages in the movie (`iso_639` attribute).
- `Keywords_names` : List of name attribute for each Keyword.
- `cast_len` : Number of members in the cast.
- `crew` : We created a column for each department in the dataset(there are 12).
In each column you can find the numbers of members in the crew from the fitting department.
- `crew_directors_names` - A column that contains a tuple of each of the director's names


``` python 
flatten_train_df = flatten_features(train_raw)
flatten_test_df = flatten_features(test_raw)
```

```python
flatten_train_df.head()
```

``` python
flatten_test_df.head()
```

### Our features candidates (before feature selection):

- `collection_size` : The amount of movies in the collection (1 if the movie is not part of a collection) - will derive from `belongs_to_collection` column.
- `budget` : The budget column in a normelize form (log/ min-max normalization)
- `[gener name]` : Dummy column for each gener (19 geners in the universe)
- `[language]` : Dummy column for the 5 most frequent movie's original languages in our universe
- `overview_word_count` : Amount of words in overview
- `popularity` : The popularity column in a normelize form (min-max normalization)
- `[production company id]` : Dummy column for the 5 most frequent movie's production companies
- `num_films_of_biggest_company` : The numbers of films that the biggest production company (The one with the hightes number of films) that participates in this movie has produced (From the currect universe)
- `num_films_of_biggest_country` : The numbers of films that the biggest production country (The one with the hightes number of films) that participates in this movie has produced (From the currect universe)
- `released_year` : The year that the movie was realed on
- `released_month` : The month that the movie was realed on
- `runtime` : The moview lenth in minutes, normelized (min-max normalization)
- `spoken_lang_len` : The amount of spoken languages in a film.
- `is_english_spoken` : Binary column, 1 if English is one of tthe spoken langueges and 0 otherwize.
- `tagline_char_count` : The length (characters) of a tagline.
- `title_char_count` : The length (characters) of a title.
- `sum_votes` : The sum of votes for the movie(calculated by vote_avg * vote_count), in a normelized form (min-max normalization).
- `[Keywords]` : There are ~10,000 unique keywords in train data, so we will create dummy column for each of the top 20 most-frequent keywords in the universe.
- `cast_len` : The nubmer of members in the cast, in a normelized form (min-max normalization).
- `cast_genders_ratio` : The gender ratio (`Females / Females + Males`) of cast.
- `[actor name]` : Dummy column for the 10 most frequent actors in our universe
- `[department name]` : The nubmer of members in the each department, in a normelized form (min-max normalization).
- `[director name]` : Dummy column for the 10 most frequent directors in our universe
- `avg_popularity_by_year` : Mean popularity of films in the released year of a movie.


