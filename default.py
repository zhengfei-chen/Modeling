import pandas as pd
import numpy as np
import seaborn as sns
import pingouin as pgimport scipy
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanrfrom sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix​from category_encoders import BinaryEncoder
from IPython.display import Image
import pydotplusimport matplotlib.pyplot as plt
%matplotlib inline
color = sns.color_palette()seed = 42

[O>> data = pd.read_csv('lending_club_loan_dataset.csv', low_memory=False)
>> data.head()

data.describe().round(3)

data.describe(include=[np.object])

# Checking data balance/proportion
loan = data.bad_loan.value_counts().to_frame().rename(columns={"bad_loan":"absolute"})
loan["percent"] = (loan.apply(lambda x: x/x.sum()*100).round(2))display(loan)
---
# pie chart
data.bad_loan.value_counts().plot(kind='pie', subplots=True, autopct='%1.2f%%', explode= (0.05, 0.05), startangle=80, legend=True, fontsize=12, figsize=(14,6), textprops={'color':"black"})
plt.legend(["0: paid loan","1: not paid loan"])

data.dtypes.sort_values(ascending=True)

data.dtypes.value_counts()

nulval = data.isnull().sum().to_frame().rename(columns={0:"absolute"})
nulval["percent"] = (nulval.apply(lambda x: x/x.sum())*100).round(2)
nulvar

# General statistics
def stats(x):
    print(f"Variable: {x}")
    print(f"Type of variable: {data[x].dtype}")
    print(f"Total observations: {data[x].shape[0]}")
    detect_null_val = data[x].isnull().values.any()
    if detect_null_val:
        print(f"Missing values: {data[x].isnull().sum()} ({(data[x].isnull().sum() / data[x].isnull().shape[0] *100).round(2)}%)")
    else:
        print(f"Missing values? {data[x].isnull().values.any()}")
    print(f"Unique values: {data[x].nunique()}")
    if data[x].dtype != "O":
        print(f"Min: {int(data[x].min())}")
        print(f"25%: {int(data[x].quantile(q=[.25]).iloc[-1])}")
        print(f"Median: {int(data[x].median())}")
        print(f"75%: {int(data[x].quantile(q=[.75]).iloc[-1])}")
        print(f"Max: {int(data[x].max())}")
        print(f"Mean: {data[x].mean()}")
        print(f"Std dev: {data[x].std()}")
        print(f"Variance: {data[x].var()}")
        print(f"Skewness: {scipy.stats.skew(data[x])}")
        print(f"Kurtosis: {scipy.stats.kurtosis(data[x])}")
        print("")
        # Percentiles 1%, 5%, 95% and 99%print("Percentiles 1%, 5%, 95%, 99%")
        display(data[x].quantile(q=[.01, .05, .95, .99]))
        print("")
    else:
        print(f"List of unique values: {data[x].unique()}")


# Variable vs. target chart
def target(x):
    short_0 = data[data.bad_loan == 0].loc[:,x]
    short_1 = data[data.bad_loan == 1].loc[:,x]

    a = np.array(short_0)
    b = np.array(short_1)

    np.warnings.filterwarnings('ignore')

    plt.hist(a, bins=40, density=True, color="g", alpha = 0.6, label='Not-default', align="left")
    plt.hist(b, bins=40, density=True, color="r", alpha = 0.6, label='Default', align="right")plt.legend(loc='upper right')
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()

​# Boxplot + Hist chart
def boxhist(x):
    variable = data[x]
    np.array(variable).mean()
    np.median(variable)f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.5, 2)})
    mean=np.array(variable).mean()
    median=np.median(variable)sns.boxplot(variable, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')sns.distplot(variable, ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')plt.title(x, fontsize=10, loc="right")
    plt.legend({'Mean':mean,'Median':median})
    ax_box.set(xlabel='')
    plt.show()

# Histogram
def hist(x):
    plt.hist(data[x], bins=25)
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()

# Pie chart
def pie(x):
    data[x].value_counts(dropna=False).plot(kind='pie', figsize=(6,5), fontsize=10, autopct='%1.1f%%', startangle=0, legend=True, textprops={'color':"white", 'weight':'bold'})

# Number of observations by class
obs = data[x].value_counts(dropna=False)
o = pd.DataFrame(obs)
o.rename(columns={x:"Freq abs"}, inplace=True)
o_pc = (data[x].value_counts(normalize=True) * 100).round(2)
obs_pc = pd.DataFrame(o_pc)
obs_pc.rename(columns={x:"percent %"}, inplace=True)
obs = pd.concat([o,obs_pc], axis=1)
display(obs)

# Variable vs. target chart
def target(x):
    short_0 = data[data.bad_loan == 0].loc[:,x]
    short_1 = data[data.bad_loan == 1].loc[:,x]

    a = np.array(short_0)
    b = np.array(short_1)

    np.warnings.filterwarnings('ignore')

    plt.hist(a, bins=40, density=True, color="g", alpha = 0.6, label='Not-default', align="left")
    plt.hist(b, bins=40, density=True, color="r", alpha = 0.6, label='Default', align="right")plt.legend(loc='upper right')
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()


# Boxplot + Hist chart
def boxhist(x):
    variable = data[x]
    np.array(variable).mean()
    np.median(variable)f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.5, 2)})
    mean=np.array(variable).mean()
    median=np.median(variable)sns.boxplot(variable, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')sns.distplot(variable, ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')plt.title(x, fontsize=10, loc="right")
    plt.legend({'Mean':mean,'Median':median})
    ax_box.set(xlabel='')
    plt.show()


# Bar chart
def bar(x):
    ax = data[x].value_counts().plot(kind="bar", figsize=(6,5), fontsize=10, color=sns.color_palette("rocket"), table=False)
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')plt.xlabel(x, fontsize=10)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.ylabel("Absolute values", fontsize=10)
    plt.title(x, fontsize=10, loc="right")


# Barh chart
def barh(x):
    data[x].value_counts().plot(kind="barh", figsize=(6,5), fontsize=10, color=sns.color_palette("rocket"), table=False)
    plt.xlabel("Absolute values", fontsize=10)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.ylabel(x, fontsize=10)
    plt.title(x, fontsize=10, loc="right")


# Pivot_table_mean
def pivot_mean(a, b, c):
    type_pivot_mean = data.pivot_table(
        columns=a,
        index=b,
        values=c, aggfunc=np.mean)
    display(type_pivot_mean)# Display pivot_table
    type_pivot_mean.sort_values(by=[b], ascending=True).plot(kind="bar", title=(b), figsize=(6,4),fontsize = 12)


# Pivot_table_sum
def pivot_sum(a, b, c):
    type_pivot_sum = data.pivot_table(
        columns=a,
        index=b,
        values=c, aggfunc=np.sum)
    display(type_pivot_sum)# Display pivot_table
    type_pivot_sum.sort_values(by=[b], ascending=True).plot(kind="bar", title=(b), figsize=(6,4),fontsize = 12)


# Scatter plot
def scatter(x, y):
    targets = data["bad_loan"].unique()for target in targets:
        a = data[data["bad_loan"] == target][x]
        b = data[data["bad_loan"] == target][y]plt.scatter(a, b, label=f"bad loan: {target}", marker="*")
    
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.title("abc", fontsize=10, loc="right")
    plt.legend()
    plt.show()


data.hist(figsize=(10,9), bins=12, ec="b", xlabelsize=8, ylabelsize=8, alpha=0.9, grid=False)plt.tight_layout()
plt.show()


for col in data.select_dtypes(include=["object"]).columns:
    data[col].value_counts().plot(kind="bar", color=sns.color_palette("rocket"))

    plt.xlabel("Class", fontsize=10)
    plt.xticks(rotation=90, horizontalalignment="center")
    plt.ylabel("Count", fontsize=10)
    plt.title(col, fontsize=10, loc="right")
    plt.show()


>> data.term = data.term.str.lower()
>> data.term.value_counts()

stats("grade")
bar("grade")
target("grade")

pivot_sum("home_ownership","grade","id")
boxhist("annual_inc")

stats("annual_inc")
target("annual_inc")

scatter("annual_inc","dti")
data.annual_inc.corr(dti)

hist("short_emp")
stats("short_emp")
target("short_emp")

boxhist("emp_length_num")
stats("emp_length_num")
target("emp_length_num")

pivot_mean("bad_loan", "purpose", "emp_length_num")
stats("home_ownership")
bar("home_ownership")
pie("home_ownership")

pivot_sum("bad_loan", "home_ownership", "id")
boxhist("dti")
stats("dti")
target("dti")
pivot_sum("home_ownership", "purpose", "dti")
pivot_sum("bad_loan", "grade", "dti")

stats("purpose")
barh("purpose")
pivot_sum("bad_loan", "purpose", "id")

pie("term")
target("term")
pivot_mean("term", "grade", "annual_inc")
target("last_delinq_none")
pie("last_delinq_none")
stats("last_delinq_none")
pivot_mean("bad_loan","purpose","last_delinq_none")
bar("last_major_derog_none")
stats("last_major_derog_none")
target("last_major_derog_none")

stats("revol_util")
scatter("annual_inc", "revol_util")
boxhist("revol_util")

stats("total_rec_late_fee")
target("total_rec_late_fee")
scatter("annual_inc", "total_rec_late_fee")

data.total_rec_late_fee.corr(annual_inc)
pivot_mean("bad_loan", "purpose", "total_rec_late_fee")

boxhist("od_ratio")
stats("od_ratio")
scatter("annual_inc", "od_ratio")

pivot_sum("bad_loan", "term", "od_ratio")
stats("bad_loan")

mask = np.triu(data.corr(), 1)
plt.figure(figsize=(19, 9))
sns.heatmap(data.corr(), annot=True, vmax=1, vmin=-1, square=True, cmap='BrBG', mask=mask)

bad_loan_c = pg.pairwise_corr(data, columns=['bad_loan'], method='pearson').loc[:,['X','Y','r']]
bad_loan_c.sort_values(by=['r'], ascending=False)

data_spear = data.copy()
data_spear.drop(["bad_loan"], axis=1, inplace=True)
spearman_rank = pg.pairwise_corr(data_spear, method='spearman').loc[:,['X','Y','r']]
pos = spearman_rank.sort_values(kind="quicksort", by=['r'], ascending=False).iloc[:5,:]
neg = spearman_rank.sort_values(kind="quicksort", by=['r'], ascending=False).iloc[-5:,:]
con = pd.concat([pos,neg], axis=0)
display(con.reset_index(drop=True))

mask = np.triu(data_spear.corr(method='spearman'), 1)
plt.figure(figsize=(19, 9))
sns.heatmap(data_spear.corr(method='spearman'), annot=True, vmax=1, vmin=-1, square=True, cmap='BrBG', mask=mask);

#### Data Wrangling: Cleansing and Feature Selection
data_ca = data.select_dtypes(exclude=["int64","float64"]).copy()
data_nu = data.select_dtypes(exclude=["object","category"]).copy()

fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(16, 8))
index = 0
axs = axs.flatten()for k,v in data_nu.items():
    sns.boxplot(y=k, data=data_nu, ax=axs[index], orient="h")
    index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

display(data.describe().loc[["mean","50%","std"]].loc[:,["annual_inc","revol_util","total_rec_late_fee"]])

print(data.annual_inc.describe())
boxhist("annual_inc")

# Dealing with the outliers through IQR score method
Q1 = data['annual_inc'].quantile(0.25)
Q3 = data['annual_inc'].quantile(0.75)
IQR = Q3 - Q1
data['annual_inc'] = data.annual_inc[~((data.annual_inc < (Q1 - 1.5 * IQR)) |(data.annual_inc > (Q3 + 1.5 * IQR)))]

boxhist("annual_inc")

print(int(data_nu.annual_inc.describe()[0]) - int(data.annual_inc.describe()[0]),"outliers were removed with this operation.")

print(data.revol_util.describe())
boxhist("revol_util")

# Dealing with the 5010.0 outlier
value = data.revol_util.quantile([.99999])
p = value.iloc[0]
data = data[data["revol_util"] < p]
print(data['revol_util'].describe())
boxhist("revol_util")

print(int(data_nu.revol_util.describe()[0]) - int(data.revol_util.describe()[0]),"outlier was removed with this operation.")

sns.boxplot(x=data['total_rec_late_fee'],data=data)plt.xlabel('total_rec_late_fee', fontsize=10)plt.show()

value = data.total_rec_late_fee.quantile([.989])
p = value.iloc[0]
data = data[data["total_rec_late_fee"] < p]
sns.boxplot(x=data['total_rec_late_fee'],data=data)
plt.xlabel('total_rec_late_fee', fontsize=10)
plt.show()

for col in data[["annual_inc", "total_rec_late_fee", "revol_util"]].columns:
    sns.boxplot(data[col])
    plt.show()

for column in data.columns:
    if data[column].isna().sum() != 0:
        missing = data[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ---> '{portion:.3f}%'")

data.annual_inc.value_counts(dropna=False)
boxhist("annual_inc")

data["annual_inc"] = data.annual_inc.fillna(data.annual_inc.mean())
print(f"Fillna done. Anomalies detected: {data.annual_inc.isnull().values.any()}")

data.home_ownership.value_counts(dropna=False)
bar("home_ownership")

data["home_ownership"] = data.home_ownership.fillna(data.home_ownership.value_counts().index[0])
print(f"Imputation done. Missing values: {data.home_ownership.isnull().sum()}")

data.dti.value_counts(dropna=False)
boxhist("dti")

data["dti"] = data.dti.fillna(data.dti.mean())
print(f"Fillna done. Missing values: {data.dti.isnull().values.any()}")

abs_mv = data.last_major_derog_none.value_counts(dropna=False)
pc_mv = data.last_major_derog_none.value_counts(dropna=False, normalize=True) * 100
pc_mv_df = pd.DataFrame(pc_mv)pc_mv_df.rename(columns={"last_major_derog_none":"Percent %"}, inplace=True)
abs_pc = pd.concat([abs_mv,pc_mv_df], axis=1)
abs_pc

data.drop("last_major_derog_none", axis=1, inplace=True)
print(f"All missing values are solved in the entire dataset: {data.notnull().values.any()}")

data.info()

data.drop("id", axis=1, inplace=True)

data_nu = data.select_dtypes(exclude=["object","category"]).copy()

Xnum = data_nu.drop(["bad_loan"], axis= "columns")
ynum = data_nu.bad_loan

# Identifying the predictive features using the Pearson Correlation p-value
pd.DataFrame(
    [scipy.stats.pearsonr(Xnum[col],
    ynum) for col in Xnum.columns],
    columns=["Pearson Corr.", "p-value"],
    index=Xnum.columns,
).round(4)

Xcat = data.select_dtypes(exclude=['int64','float64']).copy()
Xcat['target'] = data.bad_loan
Xcat.dropna(how="any", inplace=True)ycat = Xcat.target
Xcat.drop("target", axis=1, inplace=True)

for col in Xcat.columns:
    table = pd.crosstab(Xcat[col], ycat)
    print()
    display(table)
    _, pval, _, expected_table = scipy.stats.chi2_contingency(table)
    print(f"p-value: {pval:.25f}")


data["grade"] = data.grade.map({"A":7, "B":6, "C":5, "D":4, "E":3, "F":2, "G":1})

df_term = data.term
df_home = data.home_ownership
df_purp = data.purpose
#term
t_ohe = pd.get_dummies(df_term)
bin_enc_term = BinaryEncoder()
t_bin = bin_enc_term.fit_transform(df_term)
#home_ownsership
h_ohe = pd.get_dummies(df_home)
bin_enc_home = BinaryEncoder()
h_bin = bin_enc_home.fit_transform(df_home)
#purpose
p_ohe = pd.get_dummies(df_purp)
bin_enc_purp = BinaryEncoder()
p_bin = bin_enc_purp.fit_transform(df_purp)


data = pd.get_dummies(data, columns=["term","home_ownership"])
>> bin_enc_purp = BinaryEncoder()
>> data_bin = bin_enc_purp.fit_transform(data.purpose)

# Concatenating both datasets
df = pd.concat([data,data_bin],axis=1)
# Dropping 'purpose'
df.drop(["purpose"], axis=1, inplace=True)
# Lowering upper characters
df.columns = [x.lower() for x in df.columns]
# printing 5 first rows
df.head()

# ROC Curve: Area Under the Curvedef auc_roc_plot(y_test, y_preds):
    fpr, tpr, thresholds = roc_curve(y_test,y_preds)
    roc_auc = auc(fpr, tpr)    print(roc_auc)    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--'
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate'
    plt.xlabel('False Positive Rate')
    plt.show()


# Making a copy of the dataset
df_lr = df.copy()
---
# Dividing the dataset in train (80%) and test (20%)
train_set_lr, test_set_lr = train_test_split(df_lr, test_size = 0.2, random_state = seed)
X_train_lr = train_set_lr.drop(['bad_loan'], axis = 1)
y_train_lr = train_set_lr['bad_loan']
X_test_lr = test_set_lr.drop(['bad_loan'], axis = 1)
y_test_lr = test_set_lr['bad_loan']
---
# Normalizing the train and test data
scaler_lr = MinMaxScaler()
features_names = X_train_lr.columns
X_train_lr = scaler_lr.fit_transform(X_train_lr)
X_train_lr = pd.DataFrame(X_train_lr, columns = features_names)
X_test_lr = scaler_lr.transform(X_test_lr)
X_test_lr = pd.DataFrame(X_test_lr, columns = features_names)
---
%%time
lr = LogisticRegression(max_iter = 1000, solver = 'lbfgs', random_state = seed, class_weight = 'balanced' )
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
clf_lr = GridSearchCV(lr, parameters, cv = 5).fit(X_train_lr, y_train_lr)
>>> CPU times: user 10.3 s, sys: 449 ms, total: 10.8 s
Wall time: 3.21 s
---
clf_lr

y_preds_lr = clf_lr.predict_proba(X_test_lr)[:,1]
auc_roc_plot(y_test_lr, y_preds_lr)

# Confusion Matrix display
plot_confusion_matrix(clf_lr, X_test_lr, y_test_lr, values_format=".4g", cmap="Blues");
---
# Creating assignments for Final Results
tn, fp, fn, tp = confusion_matrix(y_test_lr == 1, y_preds_lr > 0.5).ravel()
tn_lr = tn
fp_lr = fp
fn_lr = fn
tp_lr = tp


#### K-Nearest Neighbors (KNN)
# Making a copy of the dataset
df_knn = df.copy()
---
# Dividing the dataset in train (80%) and test (20%)
train_set_knn, test_set_knn = train_test_split(df_knn, test_size = 0.2, random_state = seed)
​
X_train_knn = train_set_knn.drop(['bad_loan'], axis = 1)
y_train_knn = train_set_knn['bad_loan']
X_test_knn = test_set_knn.drop(['bad_loan'], axis = 1)
y_test_knn = test_set_knn['bad_loan']
---
# Normalizing train and test data
scaler_knn = MinMaxScaler()
features_names = X_train_knn.columns
X_train_knn = scaler_knn.fit_transform(X_train_knn)
X_train_knn = pd.DataFrame(X_train_knn, columns = features_names)
X_test_knn = scaler_knn.transform(X_test_knn)
X_test_knn = pd.DataFrame(X_test_knn, columns = features_names)
---
%%time
for k in range(1, 200, 5):
    k = k + 1
    knn = KNeighborsClassifier(n_neighbors = k).fit(X_train_knn, y_train_knn)
    acc = knn.score(X_test_knn, y_test_knn)
    print('Accuracy for k =', k, ' is:', acc)

%%time
knn = KNeighborsClassifier(n_neighbors = 47, weights='uniform').fit(X_train_knn, y_train_knn)
y_preds_knn = knn.predict(X_test_knn)

auc_roc_plot(y_test_knn, y_preds_knn)

# Confusion Matrix display
plot_confusion_matrix(knn, X_test_knn, y_test_knn, values_format=".4g", cmap="Blues");
---
​
# Creating assignments for Final Results
tn, fp, fn, tp = confusion_matrix(y_test_knn == 1, y_preds_knn > 0.5).ravel()
tn_knn = tn
fp_knn = fp
fn_knn = fn
tp_knn = tp

#### Support Vector Machine (SVC)
# Making a copy of the dataset
df_svm = df.copy()
---
# Dividing the dataset in train (80%) and test (20%)
train_set_svc, test_set_svc = train_test_split(df_svm, test_size = 0.2, random_state = seed)
X_train_svc = train_set_svc.drop(['bad_loan'], axis = 1)
y_train_svc = train_set_svc['bad_loan']
X_test_svc = test_set_svc.drop(['bad_loan'], axis = 1)
y_test_svc = test_set_svc['bad_loan']
---
# Standardization of train and test data
zscore_svc = StandardScaler()
features_names = X_train_svc.columns
X_train_svc = zscore_svc.fit_transform(X_train_svc)
X_train_svc = pd.DataFrame(X_train_svc, columns = features_names)
X_test_svc = zscore_svc.transform(X_test_svc)
X_test_svc = pd.DataFrame(X_test_svc, columns = features_names)
---
%%time
svc = SVC(random_state=seed, class_weight='balanced',probability=True, verbose=True)
parameters = {'C':[0.1, 1, 10]}
clf_svc = GridSearchCV(svc, parameters, cv = 5).fit(X_train_svc, y_train_svc)

%%time
y_preds_svc = clf_svc.predict_proba(X_test_svc)[:,1]
auc_roc_plot(y_test_svc, y_preds_svc)

# Confusion Matrix display
plot_confusion_matrix(clf_svc, X_test_svc, y_test_svc, values_format=".4g", cmap="Blues");
---​
# Creating assignments for Final Results
tn, fp, fn, tp = confusion_matrix(y_test_svc == 1, y_preds_svc > 0.5).ravel()
tn_svc = tn
fp_svc = fp
fn_svc = fn
tp_svc = tp


#### Decision Trees (DT)
# Making a copy of the datasetdf_trees = df.copy()---# Dividing the dataset in train (80%) and test (20%)train_set_dt, test_set_dt = train_test_split(df_trees, test_size = 0.2, random_state = seed)X_train_dt = train_set_dt.drop(['bad_loan'], axis = 1)
y_train_dt = train_set_dt['bad_loan']X_test_dt = test_set_dt.drop(['bad_loan'], axis = 1)
y_test_dt = test_set_dt['bad_loan']---%%timeclf_tree = tree.DecisionTreeClassifier(random_state = seed, max_depth = 10).fit(X_train_dt, y_train_dt)

clf_tree.score(X_test_dt, y_test_dt)

# Visualizing variables by importanceimportant_features = pd.DataFrame(data = clf_tree.feature_importances_, index = X_train_dt.columns, columns = ["value"])important_features.sort_values(by = "value", ascending = False)


y_preds_dt = clf_tree.predict_proba(X_test_dt)[:,1]---auc_roc_plot(y_test_dt, y_preds_dt)


# Confusion Matrix displayplot_confusion_matrix(clf_tree, X_test_dt, y_test_dt, values_format=".4g", cmap="Blues");---# Creating assignments Final Resultstn, fp, fn, tp = confusion_matrix(y_test_dt == 1, y_preds_dt > 0.5).ravel()tn_dt = tn
fp_dt = fp
fn_dt = fn
tp_dt = tp


#### Random Forest (RF)
# Making a copy of the datasetdf_rf = df.copy()---# Dividing the dataset in train (80%) and test (20%)train_set_rf, test_set_rf = train_test_split(df_rf, test_size = 0.2, random_state = seed)X_train_rf = train_set_rf.drop(['bad_loan'], axis = 1)
y_train_rf = train_set_rf['bad_loan']X_test_rf = test_set_rf.drop(['bad_loan'], axis = 1)
y_test_rf = test_set_rf['bad_loan']---%%timerf = RandomForestClassifier(random_state = seed, class_weight = None).fit(X_train_rf, y_train_rf)parameters = {'n_estimators':[10, 100, 300, 1000]}clf_rf = GridSearchCV(rf, parameters, cv = 5).fit(X_train_rf, y_train_rf)

y_preds_rf = clf_rf.predict_proba(X_test_rf)[:,1]

# Confusion Matrxi displayplot_confusion_matrix(clf_rf, X_test_rf, y_test_rf, values_format=".4g", cmap="Blues");---​# Creating assignments for Final Resultstn, fp, fn, tp = confusion_matrix(y_test_rf == 1, y_preds_rf > 0.5).ravel()tn_rf = tn
fp_rf = fp
fn_rf = fn
tp_rf = tp


#### Neural Networks (NN)

# Making a copy of the datasetdf_nn = df.copy()---# Dividing the dataset in train (80%) and test (20%)train_set_nn, test_set_nn = train_test_split(df_nn, test_size = 0.2, random_state = seed)X_train_nn = train_set_nn.drop(['bad_loan'], axis = 1)
y_train_nn = train_set_nn['bad_loan']X_test_nn = test_set_nn.drop(['bad_loan'], axis = 1)
y_test_nn = test_set_nn['bad_loan']---# Normalization of the train and test datascaler_nn = MinMaxScaler()
features_names = X_train_nn.columnsX_train_nn = scaler_nn.fit_transform(X_train_nn)
X_train_nn = pd.DataFrame(X_train_nn, columns = features_names)X_test_nn = scaler_nn.transform(X_test_nn)
X_test_nn = pd.DataFrame(X_test_nn, columns = features_names)---%%timemlp_nn = MLPClassifier(solver = 'adam', random_state = seed, max_iter = 1000 )parameters = {'hidden_layer_sizes': [(20,), (20,10), (20, 10, 2)], 'learning_rate_init':[0.0001, 0.001, 0.01, 0.1]}clf_nn = GridSearchCV(mlp_nn, parameters, cv = 5).fit(X_train_nn, y_train_nn)


y_preds_nn = clf_nn.predict_proba(X_test_nn)[:,1]

# Confusion Matrix displayplot_confusion_matrix(clf_nn, X_test_nn, y_test_nn, values_format=".4g", cmap="Blues");​---# Creating assignments for Final Resultstn, fp, fn, tp = confusion_matrix(y_test_nn == 1, y_preds_nn > 0.5).ravel()tn_nn = tn
fp_nn = fp
fn_nn = fn
tp_nn = tp

#### Results: Performance comparison between models
# Creating performance table
results_1 = {'Classifier': ['AUC ROC (%)','TN (%)','FP (%)','FN (%)','TP (%)'],'Logistic Regression (LR)': [aucroclr, (tn_lr/3956*100).round(2), (fp_lr/3956*100).round(2), (fn_lr/3956*100).round(2), (tp_lr/3956*100).round(2)],'K Nearest Neighbour (KNN)': [aucrocknn, (tn_knn/3956*100).round(2),(fp_knn/3956*100).round(2), (fn_knn/3956*100).round(2),(tp_nn/3956*100).round(2)],
        'Support Vector Machine (SVC)': [aucrocsvc, (tn_svc/3956*100).round(2),(fp_svc/3956*100).round(2), (fn_svc/3956*100).round(2),(tp_svc/3956*100).round(2)],
        'Decision Trees (DT)': [aucrocdt, (tn_dt/3956*100).round(2), (fp_dt/3956*100).round(2), (fn_dt/3956*100).round(2),(tp_dt/3956*100).round(2)],
        'Random Forest (RF)': [aucrocrf, (tn_rf/3956*100).round(2), (fp_rf/3956*100).round(2), (fn_rf/3956*100).round(2),(tp_rf/3956*100).round(2)],
        'Neural Networks (NN)': [aucrocnn, (tn_nn/3956*100).round(2), (fp_nn/3956*100).round(2),(fn_nn/3956*100).round(2),(tp_nn/3956*100).round(2)]}
df1 = pd.DataFrame(results_1, columns = ['Classifier', 'Logistic Regression (LR)', 'K Nearest Neighbour (KNN)', 'Support Vector Machine (SVC)', 'Decision Trees (DT)', 'Random Forest (RF)', 'Neural Networks (NN)'])
df1.set_index("Classifier", inplace=True)
results = df1.T
results

# Creating table for graphic visualizationresults_2 = {'Classifier': ['ROC AUC'], 'Logistic Regression (LR)': [aucroclr], 'K Nearest Neighbour (KNN)': [aucrocknn], 'Support Vector Machine (SVC)': [aucrocsvc], 'Decision Trees (DT)': [aucrocdt], 'Random Forest (RF)': [aucrocrf], 'Neural Networks (NN)': [aucrocnn]}df2 = pd.DataFrame(results_2, columns = ['Classifier', 'Logistic Regression (LR)', 'K Nearest Neighbour (KNN)', 'Support Vector Machine (SVC)', 'Decision Trees (DT)', 'Random Forest (RF)', 'Neural Networks (NN)'])df2.set_index("Classifier", inplace=True)
results_2 = df2---# Display tHe graphax = results_2.plot(kind="bar", title=("Evaluating models' performance"), figsize=(12,8) ,fontsize=10, grid=True)for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')plt.legend(fontsize=8.5, loc="upper right")
plt.xlabel('')
plt.xticks(rotation='horizontal')
plt.ylabel('Relative frequency (%)')
plt.show()


Conclusion

Best model: Support Vector Machine - Classifier (SVC): 75.21%.

The rule of thumb is very straightforward: the higher the value of the ROC AUC metric, the better. If a random model would show 0.5, a perfect model would achieve 1.0.

The academic scoring system stands as follows:

.9 -  1 = excellent  (A)
.8 - .9 = good       (B)
.7 - .8 = reasonable (C)
.6 - .7 = weak       (D)
.5 - .6 = terrible   (F)

The ratio between TPR and FPR determined by a threshold over which results in a positive instance puts the chosen model (SVC) at a reasonable level ( C ) with a ROC AUC score of 75.21%.





