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

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\wine-quality-white-and-red.csv")
# label_encoder_month = LabelEncoder()
# data["arrival_date_month_encoded"] = label_encoder_month.fit_transform(data["arrival_date_month"])
X = data[["fixed acidity", "volatile acidity", "citric acid","residual sugar","chlorides","free sulfur dioxide",
          "total sulfur dioxide","density","pH","sulphates","alcohol"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["type"]) #Target
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
    s.add( f >= 0, f <= 75)
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


header = list(X.columns) + ["type"]

# write dataset to csv file
import csv
with open('wine2.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
# Fixed Acidity constraint: 3.5 < Fixed Acidity < 16
    if row['fixed acidity'] < 3.5 or row['fixed acidity'] > 16:
            return True
    
    # Volatile Acidity constraint: 0.05 < Volatile Acidity < 1.6
    if row['volatile acidity'] < 0.05 or row['volatile acidity'] > 1.6:
            return True
        
    # Citric Acid constraint: 0.0 < Citric Acid < 1.7
    if row['citric acid'] < 0 or row['citric acid'] > 1.7:
            return True
        
    # Residual Sugar constraint: 0.5 < Residual Sugar < 70
    if row['residual sugar'] < 0.5 or row['residual sugar'] > 70:
            return True
        
    # Chlorides constraint: 0.0 < Chlorides < 0.65
    if row['chlorides'] < 0 or row['chlorides'] > 0.65:
            return True
        
    # Free Sulfur Dioxide constraint: 0.5 < Free Sulfur Dioxide < 300
    if row['free sulfur dioxide'] < 0.5 or row['free sulfur dioxide'] > 300:
            return True
    
    # Total Sulfur Dioxide constraint: 5 < Total Sulfur Dioxide < 450
    if row['total sulfur dioxide'] < 5 or row['total sulfur dioxide'] > 450:
            return True
    
    # Density constraint: 0.98 < Density < 1.05
    if row['density'] < 0.98 or row['density'] > 1.05:
            return True
    
    # pH constraint: 2.7 < pH < 4.1
    if row['pH'] < 2.7 or row['pH'] > 4.1:
            return True
    
    # Sulphates constraint: 0.1 < Sulphates < 2.1
    if row['sulphates'] < 0.1 or row['sulphates'] > 2.1:
            return True
    
    # Alcohol constraint: 7% < Alcohol < 15%
    if row['alcohol'] < 7 or row['alcohol'] > 15:
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