from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from z3 import *

# Add the ML-DIFF root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mldiff import dt2smt
from mldiff import svm2smt

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\diabetes.csv")
X = data[["Glucose", "BloodPressure", "Insulin", "DiabetesPedigreeFunction","Age"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Outcome"]) #Target
class_names = label_encoder.classes_
# y = data[["Outcome"]] #Target
# class_names = ['0','1']
# encoder = LabelEncoder()
# columns_to_encode = ['country', 'gender']
# for column in columns_to_encode:
#     data[column] = encoder.fit_transform(data[column])
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# train decision tree
dt = DecisionTreeClassifier()
dt.fit(X, y)

# train svm
svm = LinearSVC()
svm.fit(X, y)


# convert dt and svm to SMT
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

features = []
for i in range(X.shape[1]):
    f = Real("x" + str(i))
    features.append(f)
    # # set min and max values for the features
    # min_val = min(X[:,i])
    # max_val = max(X[:,i])
    s.add( f >= 0, f <= 100)
    # force feature to have only one decimal place
    # s.add(10*f == ToInt(10*f))
    # force feature to be an integer
    s.add(f == ToInt(f))

def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()   
    yield from all_smt_rec(list(initial_terms))

# disagreement dataset
dataset = []

s.add(clDt != clSvm)
if s.check() == sat:
    print('The two classifiers disagree on the following model:')
    # get the first 100 data instances
    for m in all_smt(s, features):
        data = []
        for f in features:
            data.append(float(s.model().eval(f, model_completion=True).as_fraction()))
        # add class label of the decision tree as a pthon int        
        data.append(s.model().eval(clDt, model_completion=True).as_long())
        dataset.append(data)
        # stop after 10000 elements in the dataset
        if len(dataset) >= 10000:
            break

print(dataset)
# replace class labels with class names
for row in dataset:
    row[-1] = y[row[-1]]

header = list(X.columns) + ["Outcome"]


# write dataset to csv file
import csv
with open('diabetes2.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)

# Function to check if a row satisfies the constraints
def is_outlier(row):
    # Glucose constraint: 0 < Glucose < 200
    if row['Glucose'] < 0 or row['Glucose'] > 200:
        return True

    # Insulin constraint: 0 < Insulin < 900
    if row['Insulin'] < 0 or row['Insulin'] > 900:
        return True

    #Blood pressure constraint: should be < 180 
    if row['BloodPressure'] > 180:
        return True
        
    #DiabetesPedigreeFunction constraint: cant be > 2.5
    if row['DiabetesPedigreeFunction'] > 2.5:
        return True
        
    #DiabetesPedigreeFunction constraint: If age >60, DiabetesPedigreeFunction < 2
    if row['Age'] > 60 and row['DiabetesPedigreeFunction'] > 2:
        return True
        
    #Outcome constraint: should be either 1 or 0
    if row['Outcome'] not in [0, 1]:
        return True

    # If all constraints are satisfied, return True
    return False

print(header)

# determine the number of elements of dataset that satisfy the constraints
count = 0
for row in dataset:
    row_dict = dict(zip(header, row))
    if is_outlier(row_dict):
        count += 1

print(f"Number of elements that` are outliers: {count} out of {len(dataset)}")