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

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\data.csv")
# label_encoder_month = LabelEncoder()
# data["arrival_date_month_encoded"] = label_encoder_month.fit_transform(data["arrival_date_month"])
X = data[["radius_mean", "perimeter_mean", "area_mean","concavity_mean","concave points_mean","perimeter_worst","area_worst"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["diagnosis"]) #Target
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
    s.add( f >= 0, f <= 150)
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


header = list(X.columns) + ["diagnosis"]

# write dataset to csv file
import csv
with open('breastcancer3.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
# Radius Mean constraint: If Malignant, radius_mean should be > 10
    if row['diagnosis'] == 'M' and row['radius_mean'] < 10:
        #print(f"Invalid radius mean: {row['radius_mean']}")
        return True

    # Radius Mean constraint: If Benign, radius_mean should be > 6
    if row['diagnosis'] == 'B' and row['radius_mean'] < 6:
        return True
        
    # Perimeter Mean constraint: If Malignant, perimeter_mean should be > 70
    if row['diagnosis'] == 'M' and row['perimeter_mean'] < 70:
        return True

    # Perimeter Mean constraint: If Benign, perimeter_mean should be > 41
    if row['diagnosis'] == 'B' and row['perimeter_mean'] < 41:
        return True
        
    # Area Mean constraint: If Malignant, area_mean should be >= 350
    if row['diagnosis'] == 'M' and row['area_mean'] < 350:
        return True

    # Area Mean constraint: If Benign, area_mean should be >= 142
    if row['diagnosis'] == 'B' and row['area_mean'] < 142:
        return True

    # Concavity Mean constraint: concavity_mean < 0.5
    if row['concavity_mean'] > 0.5:
        return True

    # Concave Points Mean constraint: concave_points_mean > 0
    if row['concave points_mean'] < 0:
        return True

    # Worst Perimeter constraint: If Malignant, perimeter_worst > 84
    if row['diagnosis'] == 'M' and row['perimeter_worst'] < 84:
        return True

    # Worst Perimeter constraint: If Benign, perimeter_worst > 49
    if row['diagnosis'] == 'B' and row['perimeter_worst'] < 49:
        return True
    
    # Worst Area constraint: If Malignant, area_worst > 500
    if row['diagnosis'] == 'M' and row['area_worst'] < 500:
        return True

    # Worst Area constraint: If Benign, area_worst > 184
    if row['diagnosis'] == 'B' and row['area_worst'] < 184:
        return True

    # If all constraints are satisfied, return 1
    return False

print(header)

# determine the number of elements of dataset that satisfy the constraints
count = 0
for row in dataset:
    row_dict = dict(zip(header, row))
    if is_outlier(row_dict):
        count += 1

print(f"Number of elements that` are outliers: {count} out of {len(dataset)}")