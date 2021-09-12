import pickle
import pandas as pd
import os


root_path =  os.getcwd()
modelPath = os.path.join(root_path, "text_classification_model.sav")
loaded_model = pickle.load(open(modelPath, 'rb'))

test_csv_file = os.path.join(root_path, "test_set.csv")
data = pd.read_csv(test_csv_file, error_bad_lines=False, encoding='iso-8859-1')
X_test = data['text'].tolist()
result = loaded_model.predict(X_test)
dictionary = {"text": X_test, "predicted_labels": result}
df = pd.DataFrame(dictionary)

df.to_csv(os.path.join(root_path, "result.csv"))
print(result)
