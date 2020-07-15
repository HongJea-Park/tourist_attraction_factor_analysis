import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression


def squared_error_info(reg, x, y):

    y_pred = reg.predict(x)
    error = y - y_pred
    df1 = x.shape[1]
    df2 = x.shape[0] - x.shape[1] - 1
    sse = sum(error ** 2)
    mse = sse / df2
    ssr = sum((y_pred - np.mean(y)) ** 2)
    msr = ssr / df1

    return df1, df2, sse, mse, ssr, msr


def coef_test(reg, x, y):

    _, _, _, mse, _, _ = squared_error_info(reg, x, y)
    n, p = x.shape
    xmat = x.values
    xmat = np.concatenate((np.ones((n, 1)), xmat), axis=1)
    xtx = np.matmul(xmat.T, xmat)
    inv_xtx = np.linalg.inv(xtx)
    h = np.matmul(inv_xtx, xmat.T)
    beta = np.matmul(h, y.values.reshape(-1, 1))

    def star(p_value):

        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ' '

    coef = {'beta': [],
            'se(beta)': [],
            't': [],
            'p-value': [],
            '0.05 <': [],
            'star': []}

    for i in range(p+1):

        t = (beta[i] / np.sqrt(inv_xtx[i, i] * mse))[0]
        p_value = ((1-scipy.stats.t.cdf(np.abs(t), n-p-1))*2)

        coef['beta'].append(beta[i][0])
        coef['se(beta)'].append(np.sqrt(inv_xtx[i, i] * mse))
        coef['t'].append(t)
        coef['p-value'].append(p_value)
        coef['0.05 <'].append(p_value < 0.05)
        coef['star'].append(star(p_value))

    coef = pd.DataFrame(coef)

    coef_index = {}
    for i in range(p+1):
        if i == 0:
            coef_index[i] = 'intercept'
        else:
            coef_index[i] = x.columns.values[i-1]

    coef = coef.rename(index=coef_index)

    return coef


l_list = [25000, 50000, 100000]
d_list = np.arange(250, 2001, 250)
c_list = [(level, distance) for level in l_list for distance in d_list]

result_list = []

temperature_file = '../results/dummy_arranged_temperature_final.csv'
sensus_file = '../results/one_to_one_geo_distance/one_to_one_geo_distance_'

temperature = pd.read_csv(temperature_file)
temperature.date = temperature.date.astype(str)
temperature_columns = {'Average_degree': 'average_degree',
                       'Average_humidity': 'average_humidity'}
temperature = temperature.rename(columns=temperature_columns)
temperature = temperature.drop(['average_degree'], axis=1)

for level, distance in c_list:

    monthly_mean_df = pd.read_csv('../results/monthly_mean_df.csv')
    sensus_info = pd.read_csv(f'{sensus_file}{level}_{distance}.csv')

    sensus_info.sensus = sensus_info.sensus.astype(str)
    sensus_info.date = sensus_info.date.astype(str)
    monthly_mean_df.date = monthly_mean_df.date.astype(str)

    spot_count = sensus_info['sensus'].value_counts()/29
    sensus_info = pd.merge(
        left=sensus_info,
        right=spot_count,
        how='left',
        left_on='sensus',
        right_index=True)
    sensus_info_columns = {'sensus_x': 'sensus',
                           'sum': 'monthly_sum',
                           'sensus_y': 'spot_count',
                           'spot_y': 'spot'}
    sensus_info = sensus_info.rename(columns=sensus_info_columns)
    sensus_info = sensus_info.drop(
        ['mean', 'log', 'new_spot', 'sum+1'], axis=1)

    count_col = ['date', 'spot']
    merge_col = ['sensus', 'date']
    merge_df = pd.merge(
        monthly_mean_df, sensus_info, how='left',
        left_on=count_col, right_on=count_col)
    merge_df = merge_df.groupby(merge_col)
    factor_cols = [
        '경험', '입장/대기시간', '박물관/전시', '도시경관', '전통건축',
        '교통/이동수단', '안내서비스', '자연경관/산책로', '쇼핑', '먹거리']
    mean_cols = [
        'monthly_sum', 'category_popularity', 'spot_popularity', 'spot_num',
        'spot_count', 'bus_station_num', 'subway_station_num']
    min_cols = ['bus_min_distance', 'subway_min_distance']
    factor_df = merge_df[factor_cols].mean().reset_index()
    mean_df = merge_df[mean_cols].mean().reset_index()
    min_df = merge_df[min_cols].min().reset_index()

    merge_df = pd.merge(
        factor_df, mean_df, how='left', left_on=merge_col, right_on=merge_col)
    merge_df = pd.merge(
        merge_df, min_df, how='left', left_on=merge_col, right_on=merge_col)
    merge_df = pd.merge(
        merge_df, temperature, how='left', left_on=['date'], right_on=['date'])

    merge_df['monthly_sum'] = np.log(merge_df['monthly_sum']+1)

    y = merge_df['monthly_sum']
    x = merge_df[merge_df.columns[2:].drop('monthly_sum')]

    alphas = [10**(-i) for i in range(1, 11)]
    reg = RidgeCV(alphas, cv=10).fit(x, y)
    # reg = LinearRegression().fit(x, y)
    t_test = coef_test(reg, x, y)

    s = t_test.loc[monthly_mean_df.columns[2:]]['0.05 <'].sum()
    result = {
        'level': level,
        'distance': distance,
        'score': reg.score(x, y),
        's': t_test.loc[monthly_mean_df.columns[2:]]['0.05 <'].sum(),
        't_test': t_test,
        'model': reg}
    result_list.append(result)
