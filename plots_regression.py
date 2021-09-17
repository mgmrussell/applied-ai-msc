from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
import constant
import numpy as np


def initialize_data():
    result_df = pd.read_csv('reg_result0p2.csv')
    std = pd.read_csv('reg_data_y_std.csv')
    stds = {}
    for _, row in std.iterrows():
        stds[row[0]] = row[1]
    result_df['target_std'] = result_df.apply(lambda x: stds[x['dataset']],   axis=1)
    result_df['mse_test_by_train'] = result_df['mse_test']/result_df['mse_train']

    result_df['unscaled_mse_train'] = result_df['mse_train']*result_df['target_std']*result_df['target_std']
    result_df['unscaled_mse_test'] = result_df['mse_test']*result_df['target_std']*result_df['target_std']

    result_df['r_mse_train'] = result_df.apply(lambda x: np.sqrt(x['mse_train']), axis=1)
    result_df['r_mse_test'] = result_df.apply(lambda x: np.sqrt(x['mse_test']), axis=1)

    result_df['ds'] = result_df.apply(lambda x: x['dataset'].split('(')[1].split(')')[0], axis=1)
    return result_df


def get_confidence_percent(t_value, df):
    sign = 1
    if (t_value < 0):
        sign = -1
        t_value = -t_value
    cis = np.linspace(0, 100, 101)
    del_t = 100
    answer = 0
    for ci in cis:
        ti = stats.t.ppf(1 - ci/100/2, 40)
        if abs(ti - t_value) < del_t:
            answer = 100 - ci
            del_t = abs(ti - t_value)
    return -sign*answer


def confidence_interval(name):
    # confidence_interval based on accuracy_test
    cols = [e.name for e in constant.Regression]
    confidence_interval = pd.DataFrame([], columns=cols)
    result_df = initialize_data()
    result_df = result_df.loc[(result_df['ts'] == 0.2)]
    for e1 in constant.Regression:
        d = {}
        for e2 in constant.Regression:
            e1_des = result_df.loc[(result_df['group'] == e1.name)][name].describe()
            e2_des = result_df.loc[(result_df['group'] == e2.name)][name].describe()
            t_value = (e1_des[1] - e2_des[1]) / np.sqrt(e1_des[2]*e1_des[2]/e1_des[0] + e2_des[2]*e2_des[2]/e2_des[0])
            n = len(result_df.loc[(result_df['group'] == e1.name)][name])
            d[e2.name] = [get_confidence_percent(t_value, 2*n - 2)]
        confidence_interval = pd.concat([confidence_interval, pd.DataFrame(data=d, index=[e1.name])])
    return confidence_interval


def accuracy(name):
    cols = [e.name for e in constant.Regression]
    mse = pd.DataFrame([], columns=cols)
    result_df = initialize_data()
    result_df = result_df.loc[(result_df['ts'] == 0.2)]
    datasets = result_df['ds'].unique()
    datasets.sort()
    for ds in datasets:
        ds_df = result_df.loc[(result_df['ds'] == ds)]
        d = {}
        for _, row in ds_df.iterrows():
            accuracy = row[name]
            algo = row['group']
            d[algo] = accuracy
        mse = pd.concat([mse, pd.DataFrame(data=d, index=[ds])])
        # mse = mse.sort_values(dataset, ascending=False)
    return mse


def compare_lms(ylabel='', column='', ax=None):
    if ax:
        g = sns.boxplot(x='group', y=column, data=result_df, showmeans=True, ax=ax)
        sns.stripplot(x='group', y=column, data=result_df, color="navy", jitter=0.05, size=2.0, ax=ax)
    else:
        sns.stripplot(x='group', y=column, data=result_df, color="navy", jitter=0.05, size=2.0)
        g = sns.boxplot(x='group', y=column, data=result_df, showmeans=True, ax=ax)
    plt.title("", loc="left")
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, size=14)
    # ax.tick_params(axis='y', labelsize=13)
    # ax.tick_params(axis='x', labelsize=13)
    return g


# Plot r2_test and r2_train
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
r2_test = accuracy('r2_test')
r2_train = accuracy('r2_train')

g1 = sns.heatmap(r2_test, cmap='bone', fmt='.3f', linewidths=0.5, annot=r2_test, cbar=False, ax=ax1)
g2 = sns.heatmap(r2_train, cmap='bone', fmt='.3f', linewidths=0.5, annot=r2_train, cbar=False, ax=ax2)
for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    ax.set_xticklabels(tl, rotation=0)
plt.show()

# Plot mse_test and mse_test/mse_train
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
mse_test = accuracy('mse_test')
mse_test_by_train = accuracy('mse_test_by_train')

g1 = sns.heatmap(mse_test, cmap='bone_r', fmt='.3f', linewidths=0.5, annot=mse_test, cbar=False, ax=ax1)
g2 = sns.heatmap(mse_test_by_train, cmap='bone_r', linewidths=0.5,
                 annot=mse_test_by_train, linecolor='gray', cbar=False, ax=ax2)
for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    ax.set_xticklabels(tl, rotation=0)
plt.show()


# confidence_interval
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

confidence_interval_test = confidence_interval('r_mse_test')
# print(confidence_interval_test)
confidence_interval_train = confidence_interval('r_mse_train')
# print(confidence_interval_train)
g1 = sns.heatmap(confidence_interval_test, cmap='bone',
                 annot=confidence_interval_test, cbar=False, ax=ax1)
g2 = sns.heatmap(confidence_interval_train, cmap='bone',
                 annot=confidence_interval_train, cbar=False, ax=ax2)
for t in g1.texts:
    t.set_text(t.get_text() + "%")
for t in g2.texts:
    t.set_text(t.get_text() + "%")
for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=0)
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
plt.show()

# Scaling to Big Data
time_result = pd.read_csv('reg_time_model.csv')
ns = np.log10(np.logspace(2, 8, 20))
d = np.log10(1000)
time = {}
for group in time_result.index:
    logn = time_result.loc[(time_result.index == group)]['log(n)'][group]
    logd = time_result.loc[(time_result.index == group)]['log(d)'][group]
    Intercept = time_result.loc[(time_result.index == group)]['Intercept'][group]
    time[group] = (logn*ns + logd*d + Intercept)
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

sns.heatmap(time_result, fmt='.2f', cmap=ListedColormap(['white']), linewidths=0.5,
            linecolor='gray', cbar=False, annot=time_result, ax=ax1)
ax1.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
for group in time_result.index:
    ax2.plot(ns, time[group], '.-', label=group)
ax2.text(3.5, 9, r'$d$ = ' + str(int(10 ** d)))
ax2.set_xlabel(r'$log_{10}(n)$')
ax2.set_ylabel(r'$log_{10}(time)$')
ax2.legend()
plt.show()


# Bias Variance
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
result_df = initialize_data()
result_df = result_df.loc[(result_df['ts'] == 0.2)]
# result_df = result_df.loc[(result_df['group'] != 'OLS')]
result_df = result_df.loc[(result_df['dataset'] != 'Automobile (AM)')]
g1 = compare_lms(ylabel=r'Bias Error', column='bias', ax=ax1)
g2 = compare_lms(ylabel=r'Variance Error', column='variance', ax=ax2)
plt.show()

# heatmap Bias Variance
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
bias = accuracy('bias')
variance = accuracy('variance')
g1 = sns.heatmap(bias, cmap='bone_r', linewidths=0.5, annot=bias, linecolor='gray', cbar=False, ax=ax1)
g2 = sns.heatmap(variance, cmap='bone_r', linewidths=0.5,
                 annot=variance, linecolor='gray', cbar=False, ax=ax2)
for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    ax.set_xticklabels(tl, rotation=0)
plt.show()


# Some Average Properties
result_df = initialize_data()
result_df = result_df.loc[(result_df['ts'] == 0.2)]
result_df['Regressors'] = result_df['group']
result_df = result_df.drop('group', axis=1)
new = result_df.groupby(['Regressors']).mean()
new['R2 Test'] = new['r2_test']
new['R2 Train'] = new['r2_train']
new['MSE Test'] = new['mse_test']
new['MSE Train'] = new['mse_train']
new['CV-Score'] = new['avg_csv']
new['Bias'] = new['bias']
new['Variance'] = new['variance']
new['Training Time (s)'] = new['t_time']

new = new.drop(['r2_train', 'r2_test', 'bias', 'variance', 't_time', 'c_time',
                'avg_csv', 'min_csv', 'max_csv', 'feat_imp', 'ts',
                'mse_train', 'mse_test', 'coeffs', 'alpha', 'l1_ratio', 'gamma', 'C',
                'target_std', 'mse_test_by_train', 'unscaled_mse_train',
                'unscaled_mse_test', 'r_mse_train', 'r_mse_test'], axis=1)
new = new.sort_values('R2 Test', ascending=False)
ax = sns.heatmap(new, cmap=ListedColormap(['white']),
                 linewidths=0.5, linecolor='gray', cbar=False,  annot=new)
ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
tl = ax.get_xticklabels()
ax.set_xticklabels(tl, rotation=0)
plt.show()
