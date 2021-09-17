from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
import constant
import numpy as np


def accuracy(name):
    cols = [e.name for e in constant.Classification]
    accuracy_test = pd.DataFrame([], columns=cols)
    result_df = pd.read_csv('clas_result.csv')
    result_df['accuracy_test_by_train'] = result_df['accuracy_test']/result_df['accuracy_train']
    result_df = result_df.loc[(result_df['ts'] == 0.2)]
    datasets = result_df['dataset'].unique()
    datasets.sort()
    for ds in datasets:
        ds_df = result_df.loc[(result_df['dataset'] == ds)]
        dataset = ds.split('(')[1].split(')')[0]
        d = {}
        for _, row in ds_df.iterrows():
            accuracy = row[name]
            algo = row['group']
            d[algo] = accuracy
        accuracy_test = pd.concat([accuracy_test, pd.DataFrame(data=d, index=[dataset])])
        # accuracy_test = accuracy_test.sort_values(dataset, ascending=False)
    return accuracy_test


def confidence_interval(name):
    # confidence_interval based on accuracy_test
    cols = [e.name for e in constant.Classification]
    confidence_interval = pd.DataFrame([], columns=cols)
    result_df = pd.read_csv('clas_result.csv')
    result_df = result_df.loc[(result_df['ts'] == 0.2)]
    for e1 in constant.Classification:
        d = {}
        for e2 in constant.Classification:
            e1_des = result_df.loc[(result_df['group'] == e1.name)][name].describe()
            e2_des = result_df.loc[(result_df['group'] == e2.name)][name].describe()
            t_value = (e1_des[1] - e2_des[1]) / np.sqrt(e1_des[2]*e1_des[2]/e1_des[0] + e2_des[2]*e2_des[2]/e2_des[0])
            n = len(result_df.loc[(result_df['group'] == e1.name)][name])
            d[e2.name] = [get_confidence_percent(t_value, 2*n - 2)]
        confidence_interval = pd.concat([confidence_interval, pd.DataFrame(data=d, index=[e1.name])])
    return confidence_interval


colors = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
          'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
          'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired',
          'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn',
          'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
          'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
          'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
          'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
          'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
          'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
          'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
          'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
          'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
          'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
          'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r',
          'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno',
          'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r',
          'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
          'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
          'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r',
          'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
          'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
          'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
          'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r',
          'vlag', 'vlag_r', 'winter', 'winter_r']


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
    return sign*answer


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


# accuracies
print('Accuracies')
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

accuracy_test = accuracy('accuracy_test')
accuracy_test_by_train = accuracy('accuracy_test_by_train')

# accuracy_train = accuracy('accuracy_train')

g1 = sns.heatmap(accuracy_test, cmap='bone', fmt='.2f', linewidths=0.5, annot=accuracy_test, cbar=False, ax=ax1)
g2 = sns.heatmap(accuracy_test_by_train, cmap='bone', fmt='.2f',
                 linewidths=0.5, annot=accuracy_test_by_train, cbar=False, ax=ax2)

for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    ax.set_xticklabels(tl, rotation=0)
plt.show()


# Bias Variance
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
result_df = pd.read_csv('clas_result.csv')
result_df = result_df.loc[(result_df['ts'] == 0.2)]
g1 = compare_lms(ylabel=r'Bias Error', column='bias', ax=ax1)
g2 = compare_lms(ylabel=r'Variance Error', column='variance', ax=ax2)
plt.show()


# ROC AUC
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
accuracy_test = accuracy('auc_test')
accuracy_train = accuracy('auc_train')
g1 = sns.heatmap(accuracy_test, cmap='bone', fmt='.2f', linewidths=0.5, annot=accuracy_test, cbar=False, ax=ax1)
g2 = sns.heatmap(accuracy_train, cmap='bone', fmt='.2f', linewidths=0.5, annot=accuracy_train, cbar=False, ax=ax2)

for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    ax.set_xticklabels(tl, rotation=0)
plt.show()


# confidence_interval
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
confidence_interval_test = confidence_interval('accuracy_test')
confidence_interval_train = confidence_interval('accuracy_train')
g1 = sns.heatmap(confidence_interval_test, fmt='.0f', cmap='bone',
                 annot=confidence_interval_test, cbar=False, ax=ax1)
g2 = sns.heatmap(confidence_interval_train, fmt='.0f', cmap='bone',
                 annot=confidence_interval_train, cbar=False, ax=ax2)
for t in g2.texts:
    t.set_text(t.get_text() + "%")
for t in g1.texts:
    t.set_text(t.get_text() + "%")
for ax in [g1, g2]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=0)
    ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
plt.show()

# Some Average Properties
cols = [e.name for e in constant.Classification]
summary_mean = pd.DataFrame([], columns=cols)
result_df = pd.read_csv('clas_result.csv')
t_time_df = pd.read_csv('t_time_class.csv')
result_df = result_df.loc[(result_df['ts'] == 0.2)]
result_df['Classifiers'] = result_df['group']
result_df = result_df.drop('group', axis=1)
new = result_df.groupby(['Classifiers']).mean()
new = new.drop('t_time', axis=1)
new = new.drop('train_0_1', axis=1)
new = new.drop('test_0_1', axis=1)
new = new.drop('c_time', axis=1)
new = new.drop('ts', axis=1)
new = new.drop('feat_imp', axis=1)
new['t_time'] = t_time_df.groupby(['group']).mean()['t_time']
new['Accuracy Test'] = new['accuracy_test']
new['Accuracy Train'] = new['accuracy_train']
new['ROC AUC Test'] = new['auc_test']
new['ROC AUC Train'] = new['auc_train']
new['CV-Score'] = new['avg_csv']
new['Bias'] = new['bias']
new['Variance'] = new['variance']
new['Training Time (s)'] = new['t_time']
new = new.drop(['variance', 'bias', 'accuracy_train', 'accuracy_test', 'avg_csv',
                'min_csv', 'max_csv', 'auc_test', 'auc_train', 't_time'], axis=1)
new = new.sort_values('Accuracy Test', ascending=False)
ax = sns.heatmap(new, fmt='.2f', cmap=ListedColormap(['white']),
                 linewidths=0.5, linecolor='gray', cbar=False,  annot=new)
ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
tl = ax.get_xticklabels()
ax.set_xticklabels(tl, rotation=0)
plt.show()

# Scaling to Big Data
time_result = pd.read_csv('class_time_model.csv')
ns = np.log10(np.logspace(2, 8, 20))
d = np.log10(1000)
n_class = np.log10(2)
time = {}
for group in time_result.index:
    logn = time_result.loc[(time_result.index == group)]['log(n)'][group]
    logd = time_result.loc[(time_result.index == group)]['log(d)'][group]
    log_n_class = time_result.loc[(time_result.index == group)]['log(n_class)'][group]
    Intercept = time_result.loc[(time_result.index == group)]['Intercept'][group]
    time[group] = (logn*ns + logd*d + n_class*log_n_class + Intercept)

f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

sns.heatmap(time_result, fmt='.2f', cmap=ListedColormap(['white']), linewidths=0.5,
            linecolor='gray', cbar=False, annot=time_result, ax=ax1)
ax1.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
for group in time_result.index:
    ax2.plot(ns, time[group], '.-', label=group)
ax2.text(3.5, 9, r'$d$ = ' + str(int(10 ** d)))
ax2.text(3.5, 8.5, r'$n_{class}$ = ' + str(int(10 ** n_class)))
ax2.set_xlabel(r'$log_{10}(n)$')
ax2.set_ylabel(r'$log_{10}(time)$')
ax2.legend()
plt.show()
