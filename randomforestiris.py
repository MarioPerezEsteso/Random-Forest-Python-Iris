from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
'''Read data from CSV with Pandas'''
df = pd.read_csv("data/iris.data", header=0)
'''Split into train and test with numpy (0.75 train, 0.25 test)'''
is_train = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[is_train], df[~is_train]
'''First 4 colums are the data to train'''
features = df.columns[:4]
'''Create random forest classifier'''
clf = RandomForestClassifier(n_jobs=2)
'''Species names to factor'''
y, species = pd.factorize(train['species'])
'''Train and test'''
clf.fit(train[features], y)
preds = species.values[clf.predict(test[features])]
print pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])