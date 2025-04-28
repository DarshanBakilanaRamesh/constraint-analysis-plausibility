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

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\data_abalone.csv")
# label_encoder_month = LabelEncoder()
# data["arrival_date_month_encoded"] = label_encoder_month.fit_transform(data["arrival_date_month"])
X = data[["Length", "Diameter", "Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Sex"]) #Target
class_names = label_encoder.classes_
#y = data["Species"]
# iris = load_iris()
# X = iris.data
# y = iris.target

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
    s.add( f >= 0, f <= 19)
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
    row[-1] = class_names[row[-1]]


header = list(X.columns) + ["Sex"]

# write dataset to csv file
import csv
with open('data_abalone3.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
# Viscera weight constraint: Viscera weight must not exceed Whole weight
    if row['Viscera weight'] > row['Whole weight']:
        return True
        
    # Length Constraint: 0.05 < Length < 1
    if row['Length'] < 0.05 or row['Length'] > 1:
        return True

    # Diameter Constraint: 0.05 < Diameter < 1
    if row['Diameter'] < 0.05 or row['Diameter'] > 1:
        return True

    # Height Constraint: 0 < Height < 1.5
    if row['Height'] < 0 or row['Height'] > 1.5:
        return True

    # Whole weight Constraint: 0 < Whole weight < 3
    if row['Whole weight'] < 0 or row['Whole weight'] > 3:
        return True

    # Shucked weight Constraint: 0 < Shucked weight < 1.6
    if row['Shucked weight'] < 0 or row['Shucked weight'] > 1.6:
        return True

    # Viscera weight Constraint: 0 < Viscera weight < 1
    if row['Viscera weight'] < 0 or row['Viscera weight'] > 1:
        return True

    # Shell weight Constraint: 0 < Shell weight < 1.5
    if row['Shell weight'] < 0 or row['Shell weight'] > 1.5:
        return True

    # Measurement constraint: Measurements must be non-negative
    if row['Length'] < 0 or row['Diameter'] < 0 or row['Height'] < 0:
        return True

    # Weightconstraint: Weights must be non-negative
    if row['Whole weight'] < 0 or row['Shucked weight'] < 0 or row['Viscera weight'] < 0 or row['Shell weight'] < 0:
        return True

    #Ring constraint: Rings must be non-negative
    if row['Rings'] < 0:
        return True

    # Density constraint: Density (Whole weight / Volume) should be realistic
    volume = row['Length'] * row['Diameter'] * row['Height']
    if volume > 0 and row['Whole weight'] / volume < 1:
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