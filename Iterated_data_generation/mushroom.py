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

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\mushrooms.csv")
label_encoder = LabelEncoder()
data = data.apply(label_encoder.fit_transform)
# data["arrival_date_month_encoded"] = label_encoder_month.fit_transform(data["arrival_date_month"])
X = data[["odor", "stalk-root", "veil-color","bruises","stalk-surface-above-ring",
          "stalk-color-below-ring","ring-number","spore-print-color","gill-color"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["class"]) #Target
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
    s.add( f >= 0, f <= 50)
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


header = list(X.columns) + ["class"]

# write dataset to csv file
import csv
with open('mushrooms2.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
# Class and Odor constraint: If odor is 'pungent' (p), class is 'poisonous' (p)
    if row['odor'] == 'p' and row['class'] == 'e':
        return True

    # If odor is 'foul' (f), class is 'poisonous' (p)
    if row['odor'] == 'f' and row['class'] == 'e':
        return True

    # If odor is 'fishy' (y), class is 'poisonous' (p)
    if row['odor'] == 'y' and row['class'] == 'e':
        return True
        
    # if odor is almond (a), class is edible (e)
    if row['odor'] == 'a' and row['class'] == 'p':
        return True

    # if odor is anise (l), class is edible (e)
    if row['odor'] == 'l' and row['class'] == 'p':
        return True

    # stalk constraint: if stalk-root is equal (e), veil-color is white (w)
    if row['stalk-root'] == 'e' and row['veil-color'] != 'w':
        return True

    # Bruises and Stalk Surface constraint: If bruises is 't' (bruises present), stalk-surface-above-ring should be 'smooth' (s) and fibrous (f)
    if row['bruises'] == 't':
        if row['stalk-surface-above-ring'] != 's' and row['stalk-surface-above-ring'] != 'f':
            return True

    # Stalk constraint: If stalk-color-below-ring is yellow (y) then class is 'poisonous' (p)
    if row['stalk-color-below-ring'] == ['y'] and row['class'] == 'e':
        return True

    # Ring constraint: If ring numner is null (n) then class is 'poisonous' (p)
    if row['ring-number'] == ['n'] and row['class'] == 'e':
        return True

    # Veil-color constraint: If veil-color is yellow (y) then class is 'poisonous' (p)
    if row['veil-color'] == ['y'] and row['class'] == 'e':
        return True

    # Spore-print-color constraint: If spore-print-coloris green (r) then class is 'poisonous' (p)
    if row['spore-print-color'] == ['r'] and row['class'] == 'e':
        return True

    # Spore-print-color constraint: If spore-print-coloris buff (b) then class is edible (e)
    if row['spore-print-color'] == ['b'] and row['class'] == 'p':
        return True
        
    # Gill color constraint: If gill-color is 'green' (r), then class is 'poisonous' (p)
    if row['gill-color'] == 'r' and row['class'] == 'e':
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