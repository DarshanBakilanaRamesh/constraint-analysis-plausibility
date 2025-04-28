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

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\zoo.csv")
X = data[["feathers", "eggs", "milk","aquatic","legs","toothed","backbone","breathes","fins"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["class_type"]) #Target
class_names = label_encoder.classes_
# y = data[["class_type"]] #Target
# class_names = ['1','2','3','4','5','6','7']


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
    s.add( f >= 0, f <= 15)
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


header = list(X.columns) + ["class_type"]

# write dataset to csv file
import csv
with open('zoo3.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
# Feather Constraint: Birds have feathers
    if row['class_type'] == 2:
        if row['feathers'] != 1:
            return True

    # Egg Constraint: Bird, Fish, Amphibian lay egg
    if row['class_type'] in [2, 4, 5]:
        if row['eggs'] != 1:
            return True

    # Milk Constraint: Mammal gives milk
    if row['class_type'] == 1:
        if row['milk'] != 1:
            return True

    # Aquatic Constraint: Fishes are always aquatic
    if row['class_type'] == 4:
        if row['aquatic'] != 1:
            return True

    # Leg Constraint: Birds have 2 legs
    if row['class_type'] == 2:
        if row['legs'] != 2:
            return True

    # Tooth Constraint: Fish don't have teeth
    if row['class_type'] == [4]:
        if row['toothed'] != 1:
            return True

    # Backbone Constraint: Mammal, Bird, Reptile and Fish don't have backbone
    if row['class_type'] in [1, 2, 3, 4]:
        if row['backbone'] != 1:
            return True
            
    # Breathe Constraint: Fishes use gills instead of lungs to breath
    if row['class_type'] == [4]:
        if row['breathes'] == 1:
            return True

    # Fin Constraint: Fish have fins
    if row['class_type'] == [4]:
        if row['fins'] != 1:
            return True

    # Leg Constraint: Insects have 6 legs
    if row['class_type'] == [6]:
        if row['legs'] != 6:
            return True

    # Leg Constraint: Amphibian have 4 legs
    if row['class_type'] == [5]:
        if row['legs'] != 4:
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