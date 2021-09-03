%matplotlib inline

import pandas as pd               # package for data analysis and manipulation 
import numpy as np                # package for scientific computing on multidimensional arrays 
import matplotlib                 # package for creating visualizations
from matplotlib import pyplot as plt
import seaborn as sns             # data visualization library based on matplotlib
import sklearn                    # machine learning library
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import scipy                      # library for mathematics, science and engineering
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import collections
import zipfile
import requests
import platform

# Check Python version
print('Python', platform.python_version())

# Check the version of packages
for package in [pd, np, matplotlib, sns, sklearn, scipy, requests, platform]:
    print (package.__name__, package.__version__)
    
# Remove the max column restriction for displaying on the screen
pd.options.display.max_columns = None

url1 = 'http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip'
url2 = 'https://drive.google.com/uc?id=1dAb2OalBSblCop9UbGjiv9qnS3N-3Kuw'
file = 'raw_data.zip'
try:
    with requests.Session() as s:
        response = s.get(url1)
    open(file, 'wb').write(response.content)
    print('Sucessful download from url 1\n')
except:
    with requests.Session() as s:
        response = s.get(url2)
    open(file, 'wb').write(response.content)
    print('Sucessful download from url 2\n')

zip_file = zipfile.ZipFile(file, mode='r')
zip_file.printdir()
path = 'winequality/winequality-red.csv' 

wine_csv = zip_file.open(path, mode='r')

# print some lines of red wine file
wine_csv.readlines(300)
wine_csv.seek(0)
wine = pd.read_csv(wine_csv, sep=';')
wine_csv.close()

print('Shape of wine =', wine.shape)
print('Number of rows = {}, Number of columns = {}'.format(wine.shape[0], wine.shape[1]))
wine.head(10)

wine.columns

wine.columns = wine.columns.str.replace(' ', '_')
wine.head(1)
wine.columns.get_loc('quality'), wine.columns.get_loc('alcohol')
new_order = [11, 10] + list(range(10))
wine = wine[wine.columns[new_order]]
wine.head(1)
print('Number of row before removing duplicates =', wine.shape[0])
print('Duplicated rows:\n', wine.duplicated())
print('Number of duplicated rows =', wine.duplicated().sum())
wine.drop_duplicates(inplace=True)
wine.reset_index(drop=True, inplace=True)
print('Number of rows after removing duplicates =', len(wine))
wine_2 = wine.copy()
wine_2.drop('quality', axis=1, inplace=True)
print('Number of duplicated rows =', wine_2.duplicated().sum())
wine.to_csv('red_wine.csv', index=False)

print('Number of rows = {}, Number of columns = {}'.format(wine.shape[0], wine.shape[1]))
wine.head()
wine.info()
wine.isna()
wine.isna().sum()
wine.isna().sum().sum()
wine['quality'].value_counts(sort=False)
wine.quality.value_counts(sort=False)
wine.quality.value_counts()
wine.groupby('quality').quality.count()
wine.describe()
plt.figure(figsize=(6, 4))

ax = sns.countplot(x='quality', data=wine, color='green')
ax.set(title='HISTOGRAM OF WINE QUALITY', xlabel='', ylabel='', yticklabels=[])
ax.tick_params(left=False)
ax.set_ylim(0, 650)
for p in ax.patches:
    ax.annotate(p.get_height(), 
                xy=(p.get_x() + p.get_width() / 2, p.get_height()), 
                xytext = (0, 10),
                textcoords = 'offset points',
                ha = 'center', 
                size=10)

plt.tight_layout()

category_dic = {3:'bad', 4:'bad', 5:'bad', 6:'good', 7:'good', 8:'good'}
wine['quality2'] = wine.quality.map(category_dic)

wine.quality2.value_counts()
np.round(wine.quality2.value_counts() / len(wine) * 100, 1)
class_labels = ['bad', 'good'] # class labels for graphs
custom_palette = {'bad':'blue', 'good':'red',
                 0:'blue', 1:'red'}
plt.figure(figsize=(3.5, 4))

ax = sns.countplot(x='quality2', data=wine, palette=custom_palette)
ax.set(title='HISTOGRAM OF WINE QUALITY', xlabel='', ylabel='', yticklabels=[])
ax.tick_params(left=False)
ax.set_ylim(0, 810)
for p in ax.patches:
    ax.annotate(p.get_height(), 
                xy=(p.get_x() + p.get_width() / 2, p.get_height()), 
                xytext = (0, 10),
                textcoords = 'offset points',
                ha = 'center', 
                size=10)

plt.tight_layout()
fig, axs = plt.subplots(4, 3, figsize=(12, 10))
fig.suptitle('VIOLIN PLOTS', fontsize=15)

column_names = wine.columns[1:12]
for i, column_name in enumerate(column_names):
    sns.violinplot(x='quality2', y=column_name, data=wine, ax=axs[i//3][i%3], palette=custom_palette)
    
axs[3][2].axis('off')
fig.tight_layout()
fig.subplots_adjust(top=0.93)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6.5))

pearson_corr = wine.iloc[:,1:12].corr(method='pearson')
spearman_corr = wine.iloc[:,1:12].corr(method='spearman')

mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=0)
cmap = sns.diverging_palette(150, 275)

sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=',.2f', cmap=cmap, 
            cbar=True, cbar_kws={"shrink": .5}, square=True, linewidths=.5, 
            vmax=0.8, vmin=-0.8, center=0, ax=ax1)
ax1.set_title('PEARSON CORRELATION MATRIX')

sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=',.2f', cmap=cmap, 
            cbar=False, square=True, linewidths=.5,
            vmax=0.8, vmin=-0.8, center=0, ax=ax2)
ax2.set_title('SPEARMAN CORRELATION MATRIX')

fig.tight_layout()


X = wine.iloc[:,1:12]

y = LabelEncoder().fit_transform(wine.quality2)
np.unique(y, return_counts=True)
class_dictionary = {'bad':0, 'good':1}
y = wine.quality2.map(class_dictionary)
y.value_counts(sort=False)
feature_names = np.array(X.columns)
print('Number of features =', len(feature_names))
print(feature_names)
# X_train - X_val - X_test
# 40 - 40 - 20

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2**9, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.50, random_state=9**2, stratify=y_train)
print('Number of rows: y_train:{}, y_val:{}, y_test:{}, Total:{}'.format(len(y_train), len(y_val), len(y_test), len(y)))

print('\nDistribution by classes:')
pd.DataFrame({'train set':np.unique(y_train, return_counts=True)[1], 
              'validation set': np.unique(y_val, return_counts=True)[1], 
              'test set': np.unique(y_test, return_counts=True)[1]})
def print_outputs(X, X_train, X_val, X_test, y, y_train, y_val, y_test, clf,
                 title='CONFUSION MATRICES'):

    train_score = clf.score(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    test_score = clf.score(X_test, y_test)

    print('   - Accuracy on training set = {:.2f}'.format(train_score))
    print('   - Accuracy on validation set = {:.2f}'.format(val_score))
    print('   - Accuracy on testing set = {:.2f}'.format(test_score))
    print('   - Total Accuracy = {:.2f}\n'.format(clf.score(X, y)))    

    y_train_predicted = clf.predict(X_train)
    y_val_predicted = clf.predict(X_val)
    y_test_predicted = clf.predict(X_test)
    
    confusion = []
    confusion.append(pd.DataFrame(confusion_matrix(y_train, y_train_predicted))) # train
    confusion.append(pd.DataFrame(confusion_matrix(y_val, y_val_predicted))) # validation
    confusion.append(pd.DataFrame(confusion_matrix(y_test, y_test_predicted))) # test

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(title, fontsize=15)
    axs = [ax1, ax2, ax3]

    for i in range(3):
        sns.heatmap(confusion[i], annot=True, fmt=',.0f', cbar=False, cmap='YlGnBu', ax= axs[i])
        axs[i].set(xticklabels=class_labels, yticklabels=class_labels, xlabel='Predicted label', ylabel='True label')      

    ax1.set_title('Training accuracy = {:.2f}'.format(train_score))
    ax2.set_title('Validation accuracy = {:.2f}'.format(val_score))
    ax3.set_title('Testing accuracy = {:.2f}'.format(test_score))

    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

clf_0 = RandomForestClassifier(random_state=0)
clf_0.fit(X_train, y_train)

print_outputs(X, X_train, X_val, X_test, y, y_train, y_val, y_test, clf_0)
clf_0c = CalibratedClassifierCV(clf_0, method='sigmoid', cv='prefit')
clf_0c.fit(X_val, y_val)

print_outputs(X, X_train, X_val, X_test, y, y_train, y_val, y_test, clf_0c, title='CONFUSION MATRICES AFTER CALIBRATING')
print('Training classifier before clalibrating:')
clf_1 = RandomForestClassifier(random_state=0)
clf_1.fit(X_val, y_val)
print_outputs(X, X_val, X_train, X_test, y, y_val, y_train, y_test, clf_1, title='CONFUSION MATRICES BEFORE CALIBRATING')

print('Training classifier after calibrating:')
clf_1c = CalibratedClassifierCV(clf_1, method='sigmoid', cv='prefit')
clf_1c.fit(X_train, y_train)
print_outputs(X, X_val, X_train, X_test, y, y_val, y_train, y_test, clf_1c, title='CONFUSION MATRICES AFTER CALIBRATING')

# clf_0 = RandomForestClassifier(random_state=0)
# clf_0.fit(X_train, y_train)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

fig.suptitle('RANDOM FOREST IMPORTANCES', fontsize=15)

feature_importances = clf_0.feature_importances_
sorted_idx = feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
ax1.barh(y_ticks, feature_importances[sorted_idx], color='blue', alpha=0.8)
ax1.set_yticklabels(feature_names[sorted_idx])
ax1.set_yticks(y_ticks)
ax1.set_title("FEATURE IMPORTANCES (MDI) ON TRAINING SET")

permutation_train = permutation_importance(clf_0, X_train, y_train, n_repeats=15, random_state=7**4, n_jobs=-1)
sorted_idx = permutation_train.importances_mean.argsort()
#print(permutation_train.importances_mean[sorted_idx].T)
ax2.boxplot(permutation_train.importances[sorted_idx].T,
           vert=False, labels=feature_names[sorted_idx])
ax2.set_title('PERMUTATION IMPORTANCES ON TRAINING SET')

ax3.axis('off')

permutation_val = permutation_importance(clf_0, X_val, y_val, n_repeats=15, random_state=6**3, n_jobs=-1)
sorted_idx = permutation_val.importances_mean.argsort()

ax4.boxplot(permutation_val.importances[sorted_idx].T,
           vert=False, labels=feature_names[sorted_idx]) 

ax4.set_title('PERMUTATION IMPORTANCES ON VALIDATION SET')

fig.tight_layout()
fig.subplots_adjust(top=0.92)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
corr = spearmanr(X_train).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=90)
ax1.set_title('DENDROGRAM FOR TRAINING SET')
ax1.axhline(1, color='cyan')

corr_dendro = corr[dendro['leaves'], :][:, dendro['leaves']] #  cluster node j appears in position i in the left-to-right traversal of the leaves
mask = np.triu(np.ones_like(corr_dendro, dtype=bool), k=0) # Generate a mask for the upper triangle

sns.heatmap(corr_dendro, mask=mask, annot=True, fmt=',.2f', cmap=cmap, 
            cbar=True, cbar_kws={"shrink": .5}, square=True, linewidths=.5, 
            center=0, vmax=0.8, vmin=-0.8, ax=ax2)

ax2.set_title('CORRELATION MATRIX ON TRAINING SET (SPEARMAN)')
ax2.set_xticklabels(dendro['ivl'], rotation=90)
ax2.set_yticklabels(dendro['ivl'], rotation=0)

fig.tight_layout();

cluster_ids = hierarchy.fcluster(corr_linkage, t=1, criterion='distance')
the_dict = collections.defaultdict(list) #defaultdict is created with the values that are list.
for idx, cluster_id in enumerate(cluster_ids):
    the_dict[cluster_id].append(idx)
selected_features = [v[0] for v in the_dict.values()]
removed_features = [i for i in range(len(feature_names)) if i not in selected_features]
#removed_features = list(set(feature_names) - set(feature_names[selected_features]))

print('Number of selected features =', len(selected_features))
print('Selected features =', feature_names[selected_features].tolist())
print('Removed features =', feature_names[removed_features].tolist(), '\n')

X_sel = X.iloc[:, selected_features]
X_train_sel = X_train.iloc[:, selected_features]
X_val_sel = X_val.iloc[:, selected_features]
X_test_sel = X_test.iloc[:, selected_features]

print('Training classifier before calibrating:')
clf_2 = RandomForestClassifier(random_state=0)
clf_2.fit(X_train_sel, y_train)
print_outputs(X_sel, X_train_sel, X_val_sel, X_test_sel, y, y_train, y_val, y_test, clf_2, 
             title='CONFUSION MATRICES FOR SELECTED FEATURES BEFORE CALIBRATING')

print('Training classifier after calibrating:')
clf_2c = CalibratedClassifierCV(clf_2, method='sigmoid', cv='prefit')
clf_2c.fit(X_val_sel, y_val)
print_outputs(X_sel, X_train_sel, X_val_sel, X_test_sel, y, y_train, y_val, y_test, clf_2c,
             title='CONFUSION MATRICES FOR SELECTED FEATURES AFTER CALIBRATING')
             
X_rem = X.iloc[:, removed_features]
X_train_rem = X_train.iloc[:, removed_features]
X_val_rem = X_val.iloc[:, removed_features]
X_test_rem = X_test.iloc[:, removed_features]

print('Training classifier before calibrating:')
clf_3 = RandomForestClassifier(random_state=0)
clf_3.fit(X_train_rem, y_train)
print_outputs(X_rem, X_train_rem, X_val_rem, X_test_rem, y, y_train, y_val, y_test, clf_3, 
             title='CONFUSION MATRICES FOR REMOVED FEATURES BEFORE CALIBRATING')

print('Training classifier after calibrating:')
clf_3c = CalibratedClassifierCV(clf_3, method='sigmoid', cv='prefit')
clf_3c.fit(X_val_rem, y_val)
print_outputs(X_rem, X_train_rem, X_val_rem, X_test_rem, y, y_train, y_val, y_test, clf_3c,
             title='CONFUSION MATRICES FOR REMOVED FEATURES AFTER CALIBRATING')
             
param_grid = {
    'n_estimators' : [50, 75, 100],    # The number of trees in the forest, default=100
    'max_features' : [2, 5],           # The number of features to consider when looking for the best split, default=sqrt(n_features)
    'max_depth'    : [3, 5, 7],        # The maximum depth of the tree, default=None
    'class_weight' : [None, 'balanced', 'balanced_subsample'] # Used to associate weights with classes, default=None
}

X_train_2 = pd.concat([X_train_sel, X_val_sel], ignore_index=True)
y_train_2 = pd.concat([y_train, y_val], ignore_index=True)

clf_4 = RandomForestClassifier(random_state=0)
skf = StratifiedKFold(n_splits=3, random_state=5**5, shuffle=True)
clf_grid = GridSearchCV(clf_4, param_grid, cv=skf, n_jobs=-1)
clf_grid.fit(X_train_2, y_train_2)
print('Best parameters found by grid search are:\n', clf_grid.best_params_)
print('\nBest cross validation score =', clf_grid.best_score_)

print_outputs(X_sel, X_train_2, X_train_2, X_test_sel, y, y_train_2, y_train_2, y_test, clf_grid.best_estimator_, 
                 title='CONFUSION MATRICES FOR SELECTED FEATURES WITH HYPERPARAMETER TUNING')             
