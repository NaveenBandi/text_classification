from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os


root_path =  os.getcwd()
train_csv = os.path.join(root_path, "train_set.csv")
data = pd.read_csv(train_csv, error_bad_lines=False, encoding='iso-8859-1')
X = data['text'].tolist()
y = data['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
model = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])
model.fit(X_train, y_train)

predicted = model.predict(X_test)

print(metrics.classification_report(y_test, predicted))
filename = os.path.join(root_path, "text_classification_model.sav")
pickle.dump(model, open(filename, 'wb'))