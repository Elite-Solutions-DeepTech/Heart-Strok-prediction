# -*- coding: utf-8 -*-
"""Heart Stroke[RFNew].ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dp5qLzXQcydktpINknsXXx1Ke2yHYbVt
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import rainbow
# %matplotlib inline

from sklearn.ensemble import RandomForestClassifier

df= pd.read_csv('/content/dataset.csv')

dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

y=dataset['target'].values
x=dataset.drop(['target'],axis=1).values

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators=10)
score = cross_val_score(randomforest_classifier, x, y, cv=10)

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

alg=RandomForestClassifier()
alg.fit(x_train,y_train)
alg.score(x_train,y_train)

y_pred=alg.predict(x_test)

import matplotlib.pyplot as plt
#plt.scatter(y_pred,y_test)
#plt.plot(y_pred,y_test)
x_line=np.arange(0,350,0.1)
y_line=np.arange(0,350,0.1)
plt.plot(x_line,y_line)
plt.scatter(x_line,y_line)
plt.show()

score.mean ()