import numpy as np
import pandas as pd
from collections import Counter
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import re

print(lgb.__version__)
np.random.seed(42)

#Global parameters
TRAIN_DUMMIES_COLS = {}

def eval_or_nan(attr):
    if pd.notnull(attr) and isinstance(attr, str):
        return eval(attr)
    return None


def map_attribute(attr, attr_name):
    # if attr is not empty ([],(),{},nan)
    if attr:
        # if the attribute is a string, do eval. else return the attribute
        attr_eval = eval(attr) if isinstance(attr, str) else attr
        return tuple(map(lambda x: x.get(attr_name, None), attr_eval))
    return None


def drop_column(df, columns):
    df_dropped = df.drop(columns, axis=1)
    return df_dropped


def get_dummies_dept(set_dept, df):
    df['id'] = df['id'].astype('str')
    df_dummies = pd.DataFrame(columns=['id'] + list(sorted(set_dept)))
    index = 0
    for _, row in df.iterrows():
        row_to_insert_dict = OrderedDict()
        row_to_insert_dict['id'] = row['id']
        dept_len_dict = row['crew_department_map']
        for dept in list(sorted(set_dept)):
            if dept_len_dict and dept in dept_len_dict.keys():
                row_to_insert_dict[f'department_{dept}'] = dept_len_dict[dept]
            else:
                row_to_insert_dict[dept] = 0
        df_dummies.loc[index] = list(row_to_insert_dict.values())
        index += 1
    # print(df_dummies)
    df_with_dummies = pd.merge(df,
                  df_dummies,
                  on='id',
                  how='inner')
    return df_with_dummies


def count_each_dept(attr):
    if attr:
        len_dict = Counter(list(attr))
        return len_dict
    return None


def apply_min_max_normalization(df, columns_to_normalize):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[columns_to_normalize] = min_max_scaler.fit_transform(df[columns_to_normalize])
    return df


def nan_to_tuple(x):
    return x if x else tuple()


def get_dummies_by_threshold(threshold, df, col_name, top_k_keys=None):
    count_dict = get_frequencies_dict(df, col_name)
    if not top_k_keys:
        top_k_keys = sorted(count_dict, key=count_dict.get, reverse=True)[:threshold]
        TRAIN_DUMMIES_COLS[col_name] = list(top_k_keys)
    # df.fillna({col_name : {}}, inplace=True)
    for element in top_k_keys:
        df[f'{col_name}_{element}'] = df[col_name].apply(lambda row: 1 if element in nan_to_tuple(row) else 0)
    return df


def flatten_list(list_of_departments, flag_set=False):
    flatten = []
    for sublist in list_of_departments:
        if sublist:
            for item in list(sublist):
                flatten.append(item)
    if flag_set:
        flatten = set(flatten)
    return flatten


def get_most_frequant_key(attr, mapping_dict):
    if attr:
        return max(map(mapping_dict.get, attr))
    return None


def word_and_char_count(x, flag_split_to_words=False):
    if flag_split_to_words:
        if pd.notnull(x):
            return len(x.split(" "))
        else:
            return 0
    return len(x) if pd.notnull(x) else 0


def get_frequencies_dict(df, attr, flag_set=False):
    return Counter(flatten_list(list(df[attr].dropna()),flag_set))


def map_lenght(attr, flag_set=False):
    if attr:
        if flag_set:
            return len(set(attr))
        return len(attr)
    return 0


def get_job_names_from_crew(list_of_jobs_dicts, job):
    list_of_ids = []
    if list_of_jobs_dicts:
        for job_dict in list_of_jobs_dicts:
            if job_dict['job'] == job:
                list_of_ids.append(job_dict['name'])
    return tuple(list_of_ids)


def genders_ratio(genders):
    arr = np.array(genders)
    males = (arr == 1).sum()
    females = (arr == 2).sum()
    if males or females:
        return males / (females + males)
    return 0


def flatten_features(df, mode):
    df_work = df.copy()

    df_work['belongs_to_collection'] = df_work['belongs_to_collection'].apply(eval_or_nan)
    df_work['belongs_to_collection_ids'] = df_work['belongs_to_collection'] \
                                            .apply(lambda x: None if pd.isna(x) else x['id']).astype('Int64')
    # genres is a list, genre is a dict
    df_work['genres'] = df_work['genres'].apply(lambda genres: tuple(genre['name'] for genre in eval(genres)))

    df_work['production_companies'] = df_work['production_companies'].apply(eval_or_nan)
    df_work['production_companies_names'] = df_work['production_companies'] \
        .apply(lambda companies: map_attribute(companies, 'name'))
    # print(df_work['production_companies_ids'])
    df_work['production_companies_origin_country'] = df_work['production_companies'] \
        .apply(lambda companies: map_attribute(companies, 'origin_country'))

    df_work['production_countries_iso'] = df_work['production_countries'].apply(lambda countries: map_attribute(countries, 'iso_3166_1'))

    df_work['release_date'] = pd.to_datetime(df_work['release_date'])
    df_work['release_month'] = df_work['release_date'].dt.month
    # df_work['release_quarter'] = df_work['release_date'].dt.quarter
    df_work['release_year'] = df_work['release_date'].dt.year

    df_work['spoken_languages'] = df_work['spoken_languages'].apply(lambda langs: map_attribute(langs, 'iso_639_1'))
    df_work['spoken_languages_len'] = df_work['spoken_languages'].apply(lambda langs: map_lenght(set(langs)))

    df_work['Keywords'] = df_work['Keywords'].apply(eval_or_nan)
    df_work['Keywords_names'] = df_work['Keywords'].apply(lambda keywords: map_attribute(keywords, 'name'))

    # extract actors ids and genders
    df_work['cast'] = df_work['cast'].apply(eval_or_nan)
    df_work['cast_names'] = df_work['cast'].apply(lambda actors: map_attribute(actors, 'name'))
    df_work['cast_len'] = df_work['cast_names'].apply(lambda names: map_lenght(names))
    # gender 1 = female, gender 2 = Male, gender 0 = non gender
    df_work['cast_genders'] = df_work['cast'].apply(lambda actors: map_attribute(actors, 'gender'))  # Gender ratio

    # extract crew ids , department, directors
    df_work['crew'] = df_work['crew'].apply(eval)
    # df_work['crew_ids'] = df_work['crew'].apply(lambda crew: map_attribute(crew, 'id'))
    df_work['crew_department'] = df_work['crew'].apply(lambda crew: map_attribute(crew, 'department'))
    df_work['crew_department_map'] = df_work['crew_department'].apply(lambda departments: count_each_dept(departments))
    df_work['crew_directors_names'] = df_work['crew'].\
                                        apply(lambda crew: get_job_names_from_crew(crew, 'Director'))

    # Create dummies for department size
    if mode == 'train':
        set_of_departments = flatten_list(list(df_work['crew_department']), flag_set=True)
        TRAIN_DUMMIES_COLS['Department'] = set_of_departments
    else:
        with open('train_dummies_columns.pkl', 'rb') as f:
            dummies_columns = pickle.load(f)
        set_of_departments = dummies_columns['Department']
    df_work_dept_dummies = get_dummies_dept(set_of_departments, df_work.copy())

    # drop nested columns
    df_work_dept_dummies.drop(['crew', 'crew_department', 'cast', 'Keywords', \
                               'belongs_to_collection', 'release_date', 'production_companies', \
                               'crew_department_map'], axis=1, inplace=True)

    return df_work_dept_dummies, set_of_departments


def extract_features(df, set_of_departments, mode):
    df_work = df.copy()

    # Collection size:
    df_work['collection_size'] = df_work.groupby('belongs_to_collection_ids')['belongs_to_collection_ids'] \
        .transform('count').fillna(1).astype(int).copy()

    # Budget LogScale:
    # df_work['budget'] = df_work['budget'].transform(np.log1p)

    # overview word count
    df_work['overview_word_count'] = df_work['overview'].apply(lambda x: word_and_char_count(x, True))

    # Company with most productions:
    companies_frequencies_dict = get_frequencies_dict(df_work, 'production_companies_names')  # {company_id : company_size}
    df_work['num_films_of_biggest_company'] = df_work['production_companies_names'] \
        .apply(lambda companies: get_most_frequant_key(companies, companies_frequencies_dict)) \
        .fillna(0).astype(int)

    # Country with most films:
    countries_frequencies_dict = get_frequencies_dict(df_work, 'production_countries_iso')
    df_work['num_films_of_biggest_country'] = df_work['production_countries_iso'] \
        .apply(lambda countries_iso: get_most_frequant_key(countries_iso, countries_frequencies_dict)) \
        .fillna(0).astype(int)

    # Is english spoken column
    df_work['is_english_spoken'] = df['spoken_languages'].apply(lambda row: 1 if 'en' in row else 0)

    # tagline & title char count
    df_work['tagline_char_count'] = df_work['tagline'].apply(lambda x: word_and_char_count(x, False))
    df_work['title_char_count'] = df_work['title'].apply(lambda x: word_and_char_count(x, False))

    # sum votes
    df_work['sum_votes'] = df_work['vote_count'] * df_work['vote_average']

    # genders ratio
    df_work['cast_genders_ratio'] = df['cast_genders'].apply(genders_ratio)

    # avg popularity by year
    avg_popularity_by_year = df_work.groupby("release_year")[['popularity']].aggregate('mean') \
                                .rename(columns={'popularity': 'avg_popularity_by_year'})
    df_work = df_work.join(avg_popularity_by_year, how='left', on='release_year')

    # Create dummies by threshold:
    dummies_by_threshold_dict = {'spoken_languages': 5,
                                 'production_companies_names': 5,
                                 'Keywords_names': 20,
                                 'cast_names': 10,
                                 'crew_directors_names': 10,
                                 'genres': 19}

    if mode == 'train':
        for col, thresh in dummies_by_threshold_dict.items():
            df_work = get_dummies_by_threshold(thresh, df_work, col)
        with open('train_dummies_columns.pkl', 'wb') as f:
            pickle.dump(TRAIN_DUMMIES_COLS, f)
    else:
        with open('train_dummies_columns.pkl', 'rb') as f:
            dummies_columns = pickle.load(f)
        for col, thresh in dummies_by_threshold_dict.items():
            df_work = get_dummies_by_threshold(thresh, df_work, col, dummies_columns[col])


    df_work = drop_column(df_work, list(dummies_by_threshold_dict.keys()) + ['spoken_languages', 'original_language', \
                                                                            'overview', 'production_countries', \
                                                                            'tagline', 'title', \
                                                                            'production_companies_origin_country', \
                                                                            'production_countries_iso', 'cast_genders'])

    # Columns for normalization
    columns_to_normalize = ['popularity', 'runtime', 'cast_len', 'sum_votes', 'budget'] + list(set_of_departments)
    df_normalized = apply_min_max_normalization(df_work, columns_to_normalize)

    return df_normalized


def imputation(df):
    df_work = df.copy()
    df_work.fillna(0, inplace=True)
    df_work['budget'].fillna(0, inplace=True)
    df_work['budget'].replace(0, -1, inplace=True)

    df_work['runtime'].fillna(0, inplace=True)
    df_work['runtime'].replace(0, -1, inplace=True)

    imputer = KNNImputer(missing_values=-1)
    imputed = imputer.fit_transform(df_work)

    return pd.DataFrame(imputed, columns=df_work.columns, index=df_work.index)


def train_regression(train, label, model, model_name):
    model.fit(train, np.log1p(label))
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    predictions = np.expm1(model.predict(train))
    RMSLE = np.sqrt(mean_squared_log_error(label, predictions))
    print(f"RMSLE for train dataset for {model_name} model is: {round(RMSLE, 4)}")
    return predictions


def evaluate_regression(test, label, model_name):
    with open(f'models/{model_name}.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    if model_name == 'xgb':
        xgb.plot_importance(trained_model, max_num_features=10)
        plt.rcParams["figure.figsize"] = (25, 25)
        plt.show()
    predictions = np.expm1(trained_model.predict(test))
    RMSLE = np.sqrt(mean_squared_log_error(label, predictions))
    print(f"RMSLE for test dataset for {model_name} model is: {round(RMSLE, 4)}")
    return predictions



def regression(X, y, mode, model_name, model_params=None):
    if model_name == 'xgb':
        xgb_params = {   'subsample': 0.6,
                'reg_lambda': 10,
                'reg_alpha': 2,
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'min_child_weight': 4,
                'max_depth': 7,
                'learning_rate': 0.01,
                'gamma': 0.5,
                'colsample_bytree': 0.6 }
        model_to_train = xgb.XGBRegressor(**xgb_params, n_jobs=-1)
    elif model_name == 'lgb':
        lgb_params = {'boosting_type': 'gbdt',
                'class_weight': None,
                'colsample_bytree': 1.0,
                'importance_type': 'split',
                'learning_rate': 0.1,
                'min_child_samples': 20,
                'min_split_gain': 0.0,
                'n_jobs': -1,
                'num_leaves': 31,
                'objective': 'regression',
                'random_state': None,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'silent': True,
                'subsample': 1.0,
                'subsample_for_bin': 200000,
                'subsample_freq': 0,
                'max_depth':-1,
                'min_child_weight': 0.04,
                'n_estimators': 100}
        X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        model_to_train = lgb.LGBMRegressor(**lgb_params)
    elif model_name =='rf':
        rf_params = {'n_estimators': 1500,
                        'min_samples_split': 2,
                        'min_samples_leaf': 2,
                        'max_features': 0.4,
                        'max_depth': 50,
                        'criterion': 'mae',
                        'bootstrap': False}
        model_to_train = RandomForestRegressor(**rf_params, n_jobs=-1)
    if mode == 'train':
        predictions = train_regression(X, y, model_to_train, model_name)
    else:
        predictions = evaluate_regression(X, y, model_name)
    return predictions

def tune_params(model, params,x_train,y_train):
    tuned_model = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter = 10,
                                scoring='neg_mean_squared_log_error', verbose=2, random_state=42,
                                n_jobs=-1, return_train_score=True)

    tuned_model.fit(x_train, y_train)
    print(tuned_model.best_params_)
    return tuned_model.best_params_

def prepare_data_for_regression(df, mode):
    data = df.copy()
    cols_to_drop = ['backdrop_path', 'homepage', 'imdb_id', 'original_title', 'poster_path', 'status', 'video']
    data = drop_column(data, cols_to_drop)
    label = data['revenue']
    data = data.drop(['revenue'], axis=1)
    flatten_df, set_of_departments = flatten_features(data, mode)
    # extracting features
    df_transformed = extract_features(flatten_df, set_of_departments, mode)
    imputed_df = imputation(df_transformed)
    imputed_df = imputed_df.drop(['id'], axis=1)
    return imputed_df, label


if __name__ == '__main__':
    folder_path = 'hw1_data/'
    mode = 'test'
    df = pd.read_csv(folder_path + mode + '.tsv', sep='\t')
    X, y = prepare_data_for_regression(df, mode)
    df_presiction_results = pd.DataFrame()
    df_presiction_results['true_label'] = list(y)
    # for model in ['xgb', 'lgb', 'rf']:
    for model in ['xgb']:
        predictions = regression(X, y, mode, model)
        df_presiction_results[model] = list(predictions)


