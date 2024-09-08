import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from data_processing import split_data

def correlation_among_numeric_features(df, cols):
    numeric_col = df[cols]
    corr = numeric_col.corr()

    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs (corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features

def lr_model(x_train, y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr

def identify_significant_vars(lr, p_value_threshold = 0.05):
    print(lr.pvalues)
    print(lr.rsquared)
    print(lr.rsquared_adj)

    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var]]
    return significant_vars

if __name__ == "__main__":
    capped_data = pd.read_csv(r"D:\Cancer mortality prediction\ols regression challenge data\data\capped_data.csv")
    print(capped_data.shape)
    #cols = capped_data.nunique()[capped_data.nunique() > 3.keys().tolist()]
    #len(cols)

    corr_features = correlation_among_numeric_features(capped_data, capped_data.columns)
    print(corr_features)

    highly_corr_cols = [
        'upper_bound', 'lower_bound', 'medianagefemale', 
        'pctempprivcoverage', 'pctprivatecoverage', 'pctblack', 'pctpubliccoveragealone', 
        'povertypercent', 'medianagemale', 'pctprivatecoveragealone', 'pctmarriedhouseholds', 
        'popest2015', 'median', 'state_ District of Columbia'
    ]

cols = [col for col in capped_data.columns if col not in highly_corr_cols]
len(cols)
x_train, x_test, y_train, y_test = split_data(capped_data[cols], "target_deathrate")
lr= lr_model(x_train, y_train)
summary = lr.summary()
print(summary)

significant_vars = identify_significant_vars(lr)
print(len(significant_vars))

#train the model with significant values
significant_vars.remove('const')
x_train = sm.add_constant(x_train)
lr = lr_model(x_train[significant_vars], y_train)
summary = lr.summary()
print(summary)