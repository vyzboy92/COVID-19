from math import ceil
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import timedelta
import json
import sys


def train_model():
    try:
        df = pd.read_csv('data/covid_confirmed.csv', parse_dates=["Date"])
        X_mask_cat = ['Outbreak_enc', 'Region_enc', 'Month', 'Week']
        X_mask_lags = [c for c in df.columns if 'Confirmed_Lag_' in c]
        train_test = df.loc[df['Confirmed'] > 0].copy()
        s_unique_values = train_test[X_mask_lags].apply(lambda r: len(np.unique(r.values)), axis=1)
        train_test = train_test.loc[s_unique_values > 1].copy()
        train, valid = train_test_split(train_test, test_size=0.2, shuffle=True, random_state=231451)
        model_lgbm = lgb.LGBMRegressor(n_estimators=300, metric='mae', random_state=1234, min_child_samples=1)
        model_lgbm.fit(X=train[X_mask_cat + X_mask_lags], y=train['Confirmed'],
                       eval_set=(valid[X_mask_cat + X_mask_lags], valid['Confirmed']),
                       early_stopping_rounds=50, verbose=10)
        model_lgbm.booster_.save_model('models/lgb_conf.txt', num_iteration=model_lgbm.booster_.best_iteration)

        df = pd.read_csv('data/covid_deaths.csv', parse_dates=["Date"])
        X_mask_cat = ['Outbreak_enc', 'Region_enc', 'Month', 'Week']
        X_mask_lags = [c for c in df.columns if 'Deaths_Lag_' in c]
        train_test = df.loc[df['Deaths'] > 0].copy()
        s_unique_values = train_test[X_mask_lags].apply(lambda r: len(np.unique(r.values)), axis=1)
        train_test = train_test.loc[s_unique_values > 1].copy()
        train, valid = train_test_split(train_test, test_size=0.2, shuffle=True, random_state=231451)
        model_lgbm = lgb.LGBMRegressor(n_estimators=300, metric='mae', random_state=1234, min_child_samples=1)
        model_lgbm.fit(X=train[X_mask_cat + X_mask_lags], y=train['Deaths'],
                       eval_set=(valid[X_mask_cat + X_mask_lags], valid['Deaths']),
                       early_stopping_rounds=50, verbose=10)
        model_lgbm.booster_.save_model('models/lgb_death.txt', num_iteration=model_lgbm.booster_.best_iteration)

        df = pd.read_csv('data/covid_recovered.csv', parse_dates=["Date"])
        X_mask_cat = ['Outbreak_enc', 'Region_enc', 'Month', 'Week']
        X_mask_lags = [c for c in df.columns if 'Recovered_Lag_' in c]
        train_test = df.loc[df['Recovered'] > 0].copy()
        s_unique_values = train_test[X_mask_lags].apply(lambda r: len(np.unique(r.values)), axis=1)
        train_test = train_test.loc[s_unique_values > 1].copy()
        train, valid = train_test_split(train_test, test_size=0.2, shuffle=True, random_state=231451)
        model_lgbm = lgb.LGBMRegressor(n_estimators=300, metric='mae', random_state=1234, min_child_samples=1)
        model_lgbm.fit(X=train[X_mask_cat + X_mask_lags], y=train['Recovered'],
                       eval_set=(valid[X_mask_cat + X_mask_lags], valid['Recovered']),
                       early_stopping_rounds=50, verbose=10)
        model_lgbm.booster_.save_model('models/lgb_recovered.txt', num_iteration=model_lgbm.booster_.best_iteration)

        return True
    except Exception as e:
        return False
    

def model_predict(pred_steps):
    n_lags = 5
    final_dict = dict()
    try:
        country_dat = pd.read_csv("data/countries of the world.csv")
        country_dat['Country'] = country_dat['Country'].str.strip()
        countries = country_dat['Country'].to_list()

        # --------------Confirmed--------------

        model_lgbm = lgb.Booster(model_file='models/lgb_conf.txt')
        df = pd.read_csv('data/covid_confirmed.csv', parse_dates=["Date"])
        X_mask_cat = ['Outbreak_enc', 'Region_enc', 'Month', 'Week']
        X_mask_lags = [c for c in df.columns if 'Confirmed_Lag_' in c]

        for country in countries:
            final_dict[country] = {}
            history = df.loc[(df['Outbreak'] == 'COVID_2019') & (df['Confirmed'] > 0) & (df['Country/Region'] == str(country))]

            if history.empty is not True:
                history0 = history.iloc[-1]

                dt_rng = pd.date_range(start=history0['Date'] + timedelta(days=1),
                                       end=history0['Date'] + timedelta(days=pred_steps), freq='D').values
                dt_rng = pd.to_datetime(dt_rng)

                pred_months = pd.Series(dt_rng).apply(lambda dt: dt.month)
                pred_weeks = pd.Series(dt_rng).apply(lambda dt: dt.week)

                pred_cat = history0[X_mask_cat].values
                pred_lags = history0[X_mask_lags].values
                y = history0['Confirmed']
                pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
                pred_lags[n_lags - 1] = y
                pred = np.zeros(pred_steps)
                for d in range(pred_steps):
                    pred_cat[1] = pred_months[d]
                    pred_cat[2] = pred_weeks[d]

                    y = model_lgbm.predict(np.hstack([pred_cat, pred_lags]).reshape(1, -1))[0]

                    pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
                    pred_lags[n_lags - 1] = y  # Lag

                    pred[d] = y

                preds_conf = pd.Series(data=pred, index=dt_rng, name='LGBM predicted')
                final_dict[country]["Confirmed"] = {}
                for k, v in preds_conf.items():
                    final_dict[country]["Confirmed"][str(k)] = ceil(v)

        # ---------------- Dead ----------------------

        model_lgbm = lgb.Booster(model_file='models/lgb_death.txt')
        df = pd.read_csv("data/covid_deaths.csv", parse_dates=["Date"])
        X_mask_cat = ['Outbreak_enc', 'Region_enc', 'Month', 'Week']
        X_mask_lags = [c for c in df.columns if 'Deaths_Lag_' in c]

        for country in countries:

            history = df.loc[(df['Outbreak'] == 'COVID_2019') & (df['Deaths'] > 0) & (df['Country/Region'] == str(country))]
            if history.empty is not True:
                history0 = history.iloc[-1]

                dt_rng = pd.date_range(start=history0['Date'] + timedelta(days=1),
                                       end=history0['Date'] + timedelta(days=pred_steps), freq='D').values
                dt_rng = pd.to_datetime(dt_rng)

                pred_months = pd.Series(dt_rng).apply(lambda dt: dt.month)
                pred_weeks = pd.Series(dt_rng).apply(lambda dt: dt.week)

                pred_cat = history0[X_mask_cat].values
                pred_lags = history0[X_mask_lags].values
                y = history0['Deaths']
                pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
                pred_lags[n_lags - 1] = y
                pred = np.zeros(pred_steps)
                for d in range(pred_steps):
                    pred_cat[1] = pred_months[d]
                    pred_cat[2] = pred_weeks[d]

                    y = model_lgbm.predict(np.hstack([pred_cat, pred_lags]).reshape(1, -1))[0]

                    pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
                    pred_lags[n_lags - 1] = y  # Lag

                    pred[d] = y

                preds_conf = pd.Series(data=pred, index=dt_rng, name='LGBM predicted')
                final_dict[country]["Deaths"] = {}
                for k, v in preds_conf.items():
                    final_dict[country]["Deaths"][str(k)] = ceil(v)

        # -------------------Recovered---------------------------

        model_lgbm = lgb.Booster(model_file='models/lgb_recovered.txt')
        df = pd.read_csv("data/covid_recovered.csv", parse_dates=["Date"])
        X_mask_cat = ['Outbreak_enc', 'Region_enc', 'Month', 'Week']
        X_mask_lags = [c for c in df.columns if 'Recovered_Lag_' in c]

        for country in countries:

            history = df.loc[(df['Outbreak'] == 'COVID_2019') & (df['Recovered'] > 0) & (df['Country/Region'] == str(country))]
            if history.empty is not True:
                history0 = history.iloc[-1]

                dt_rng = pd.date_range(start=history0['Date'] + timedelta(days=1),
                                       end=history0['Date'] + timedelta(days=pred_steps), freq='D').values
                dt_rng = pd.to_datetime(dt_rng)

                pred_months = pd.Series(dt_rng).apply(lambda dt: dt.month)
                pred_weeks = pd.Series(dt_rng).apply(lambda dt: dt.week)

                pred_cat = history0[X_mask_cat].values
                pred_lags = history0[X_mask_lags].values
                y = history0['Recovered']
                pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
                pred_lags[n_lags - 1] = y
                pred = np.zeros(pred_steps)
                for d in range(pred_steps):
                    pred_cat[1] = pred_months[d]
                    pred_cat[2] = pred_weeks[d]

                    y = model_lgbm.predict(np.hstack([pred_cat, pred_lags]).reshape(1, -1))[0]

                    pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
                    pred_lags[n_lags - 1] = y  # Lag

                    pred[d] = y

                preds_conf = pd.Series(data=pred, index=dt_rng, name='LGBM predicted')
                final_dict[country]["Recovered"] = {}
                for k, v in preds_conf.items():
                    final_dict[country]["Recovered"][str(k)] = ceil(v)

        final_dict["success"] = True
        with open('data/prediction_result.json', 'w') as fp:
            json.dump(final_dict, fp)
        return final_dict
    except Exception as e:
        final_dict["success"] = False
        final_dict["error"] = str(e)
        return final_dict


def data_cleaner(df_cov):
    df_cov['Date'] = pd.to_datetime(df_cov['ObservationDate'])
    df_cov['Outbreak'] = "COVID_2019"
    temp = ['Outbreak', 'Province/State', 'Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered']
    df = df_cov[temp]
    df = df.reset_index(drop=True)
    df['Confirmed'] = df['Confirmed'].fillna(0)
    df['Deaths'] = df['Deaths'].fillna(0)
    df['Recovered'] = df['Recovered'].fillna(0)
    df['Province/State'] = df['Province/State'].fillna('Others')
    df = df.sort_values(['Country/Region', 'Province/State', 'Date'])
    df.loc[df['Country/Region'] == 'US', 'Country/Region'] = 'United States'
    df.loc[df['Country/Region'] == 'Mainland China', 'Country/Region'] = 'China'
    df.loc[df['Country/Region'] == 'Viet Nam', 'Country/Region'] = 'Vietnam'
    df.loc[df['Country/Region'] == 'UK', 'Country/Region'] = 'United Kingdom'
    df.loc[df['Country/Region'] == 'South Korea', 'Country/Region'] = 'Korea, South'
    df.loc[df['Country/Region'] == 'Taiwan, China', 'Country/Region'] = 'Taiwan'
    df.loc[df['Country/Region'] == 'Hong Kong SAR, China', 'Country/Region'] = 'Hong Kong'
    return df


def df_builder(df, country_dat):
    n_lags = 5

    # ---------------Confirmed-----------------
    df_conf = df.groupby(['Outbreak', 'Country/Region', 'Province/State', 'Date']).agg(
        {'Confirmed': 'sum'}).reset_index()
    df['Province/State'] = 'all'
    df_dead = df.groupby(['Outbreak', 'Country/Region', 'Province/State', 'Date']).agg({'Deaths': 'sum'}).reset_index()
    df['Province/State'] = 'all'
    df_rec = df.groupby(['Outbreak', 'Country/Region', 'Province/State', 'Date']).agg(
        {'Recovered': 'sum'}).reset_index()
    df['Province/State'] = 'all'

    t_conf = df_conf.groupby(['Outbreak', 'Country/Region', 'Province/State', 'Date']).agg({'Confirmed': 'max'})
    t_conf = t_conf.loc[t_conf['Confirmed'] > 0]
    df_conf = pd.merge(df_conf, t_conf[[]], left_on=['Outbreak', 'Country/Region', 'Province/State', 'Date'],
                       right_index=True)

    df_conf = pd.merge(df_conf, country_dat, how='left', left_on=['Country/Region'], right_on=['Country'])
    df_conf['Date'] = pd.to_datetime(df_conf['Date'])
    df_conf.loc[df_conf['Region'].isnull(), 'Region'] = 'Others'
    df_conf.loc[df_conf['Country'].isnull(), 'Country'] = 'Undefined'
    transformer = MinMaxScaler(feature_range=(0, 1)).fit(np.asarray([0, 2E5]).reshape(-1, 1))
    df_conf['Confirmed_transformed'] = pd.Series(
        transformer.transform(df_conf['Confirmed'].values.reshape(-1, 1)).reshape(-1))
    df_conf["Month"] = df_conf["Date"].dt.month
    df_conf["Week"] = df_conf["Date"].dt.week
    for k, v in df_conf.groupby(['Outbreak', 'Country/Region', 'Province/State']):
        for d in range(n_lags, 0, -1):
            df_conf.loc[v.index, f'Confirmed_Lag_{d}'] = v['Confirmed'].shift(d)
            df_conf.loc[v.index, f'Confirmed_Transformed_Lag_{d}'] = v['Confirmed_transformed'].shift(d)

    x_mask_lags = [c for c in df_conf.columns if
                   'Confirmed_Lag_' in c]  # + [c for c in df_conf.columns if 'Confirmed_Rolling_Mean_Lag' in c]
    x_mask_lags_transformed = [c for c in df_conf.columns if 'Confirmed_Transformed_Lag_' in c]

    df_conf[x_mask_lags] = df_conf[x_mask_lags].fillna(0)
    df_conf[x_mask_lags_transformed] = df_conf[x_mask_lags_transformed].fillna(0)
    enc_outb = LabelEncoder().fit(df_conf['Outbreak'])
    df_conf['Outbreak_enc'] = enc_outb.transform(df_conf['Outbreak'])

    enc_ctry = LabelEncoder().fit(df_conf['Country/Region'])
    df_conf['Country_enc'] = enc_ctry.transform(df_conf['Country/Region'])

    enc_region = LabelEncoder().fit(df_conf['Region'])
    df_conf['Region_enc'] = enc_region.transform(df_conf['Region'])

    # -------------Deaths--------------------

    t_dead = df_dead.groupby(['Outbreak', 'Country/Region', 'Province/State', 'Date']).agg({'Deaths': 'max'})
    t_dead = t_dead.loc[t_dead['Deaths'] > 0]
    df_dead = pd.merge(df_dead, t_dead[[]], left_on=['Outbreak', 'Country/Region', 'Province/State', 'Date'],
                       right_index=True)
    df_dead = pd.merge(df_dead, country_dat, how='left', left_on=['Country/Region'], right_on=['Country'])
    df_dead['Date'] = pd.to_datetime(df_dead['Date'])
    df_dead.loc[df_dead['Region'].isnull(), 'Region'] = 'Others'
    df_dead.loc[df_dead['Country'].isnull(), 'Country'] = 'Undefined'
    transformer = MinMaxScaler(feature_range=(0, 1)).fit(np.asarray([0, 2E5]).reshape(-1, 1))
    df_dead['Deaths_transformed'] = pd.Series(
        transformer.transform(df_dead['Deaths'].values.reshape(-1, 1)).reshape(-1))
    df_dead["Month"] = df_dead["Date"].dt.month
    df_dead["Week"] = df_dead["Date"].dt.week
    for k, v in df_dead.groupby(['Outbreak', 'Country/Region', 'Province/State']):
        for d in range(n_lags, 0, -1):
            df_dead.loc[v.index, f'Deaths_Lag_{d}'] = v['Deaths'].shift(d)
            df_dead.loc[v.index, f'Deaths_Transformed_Lag_{d}'] = v['Deaths_transformed'].shift(d)

    x_mask_lags = [c for c in df_dead.columns if
                   'Deaths_Lag_' in c]  # + [c for c in df_dead.columns if 'Deaths_Rolling_Mean_Lag' in c]
    x_mask_lags_transformed = [c for c in df_dead.columns if 'Deaths_Transformed_Lag_' in c]

    df_dead[x_mask_lags] = df_dead[x_mask_lags].fillna(0)
    df_dead[x_mask_lags_transformed] = df_dead[x_mask_lags_transformed].fillna(0)
    enc_outb = LabelEncoder().fit(df_dead['Outbreak'])
    df_dead['Outbreak_enc'] = enc_outb.transform(df_dead['Outbreak'])

    enc_ctry = LabelEncoder().fit(df_dead['Country/Region'])
    df_dead['Country_enc'] = enc_ctry.transform(df_dead['Country/Region'])

    enc_region = LabelEncoder().fit(df_dead['Region'])
    df_dead['Region_enc'] = enc_region.transform(df_dead['Region'])

    # -----------------Recovered------------------

    t_rec = df_rec.groupby(['Outbreak', 'Country/Region', 'Province/State', 'Date']).agg({'Recovered': 'max'})
    t_rec = t_rec.loc[t_rec['Recovered'] > 0]
    df_rec = pd.merge(df_rec, t_rec[[]], left_on=['Outbreak', 'Country/Region', 'Province/State', 'Date'],
                      right_index=True)
    df_rec = pd.merge(df_rec, country_dat, how='left', left_on=['Country/Region'], right_on=['Country'])
    df_rec['Date'] = pd.to_datetime(df_rec['Date'])
    df_rec.loc[df_rec['Region'].isnull(), 'Region'] = 'Others'
    df_rec.loc[df_rec['Country'].isnull(), 'Country'] = 'Undefined'
    transformer = MinMaxScaler(feature_range=(0, 1)).fit(np.asarray([0, 2E5]).reshape(-1, 1))
    df_rec['Recovered_transformed'] = pd.Series(
        transformer.transform(df_rec['Recovered'].values.reshape(-1, 1)).reshape(-1))
    df_rec["Month"] = df_rec["Date"].dt.month
    df_rec["Week"] = df_rec["Date"].dt.week
    for k, v in df_rec.groupby(['Outbreak', 'Country/Region', 'Province/State']):
        for d in range(n_lags, 0, -1):
            df_rec.loc[v.index, f'Recovered_Lag_{d}'] = v['Recovered'].shift(d)
            df_rec.loc[v.index, f'Recovered_Transformed_Lag_{d}'] = v['Recovered_transformed'].shift(d)

    x_mask_lags = [c for c in df_rec.columns if
                   'Recovered_Lag_' in c]  # + [c for c in df_rec.columns if 'Recovered_Rolling_Mean_Lag' in c]
    x_mask_lags_transformed = [c for c in df_rec.columns if 'Recovered_Transformed_Lag_' in c]

    df_rec[x_mask_lags] = df_rec[x_mask_lags].fillna(0)
    df_rec[x_mask_lags_transformed] = df_rec[x_mask_lags_transformed].fillna(0)
    enc_outb = LabelEncoder().fit(df_rec['Outbreak'])
    df_rec['Outbreak_enc'] = enc_outb.transform(df_rec['Outbreak'])

    enc_ctry = LabelEncoder().fit(df_rec['Country/Region'])
    df_rec['Country_enc'] = enc_ctry.transform(df_rec['Country/Region'])

    enc_region = LabelEncoder().fit(df_rec['Region'])
    df_rec['Region_enc'] = enc_region.transform(df_rec['Region'])

    return df_conf, df_dead, df_rec


if __name__ == "__main__":
    df_covid = pd.read_csv('data/covid_19_data.csv')
    df_covid = data_cleaner(df_covid)
    country_data = pd.read_csv("data/countries of the world.csv")
    country_data['Country'] = country_data['Country'].str.strip()

    conf, dead, rec = df_builder(df_covid, country_data)
    conf.to_csv('data/covid_confirmed.csv')
    dead.to_csv('data/covid_deaths.csv')
    rec.to_csv('data/covid_recovered.csv')
    train_model()
    dat = model_predict(int(sys.argv[1]))
    print(dat)