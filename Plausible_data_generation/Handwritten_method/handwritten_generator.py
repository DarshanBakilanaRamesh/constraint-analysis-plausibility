from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import psutil, os
from z3 import *

# SMT solver of MLDiff modified by adding hand-written constraints in order to generate more plausible results.

start_time = time.time()
process = psutil.Process(os.getpid())
# Add the ML-DIFF root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mldiff import dt2smt
from mldiff import svm2smt

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\Iris.csv")
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Species"]) #Target
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
    s.add( f >= 0, f <= 15)
    # force feature to have only one decimal place
    # s.add(10*f == ToInt(10*f))
    # force feature to be an integer
    # s.add(f == ToInt(f)) # IMPORTANT: remove constraint that we just added for the generation of dummy data

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

# adding hand-written constraints
# order of features ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
# general rule for translating handwritten checks for outliers: translate the check 
# with the corresponding variable from features[] and then negate it

#constraint for outlier was: row['SepalLengthCm'] < 4 or row['SepalLengthCm'] >8
s.add(Not(Or(features[0] < 4, features[0] > 8)))

#constraint for outlier was: row['SepalWidthCm'] < 2 or row['SepalWidthCm'] > 4.5
s.add(Not(Or(features[1] < 2, features[1] > 4.5)))

#constraint for outlier was: if row['Species'] == 'Iris-setosa':
                            #   row['PetalLengthCm'] < 1 or row['PetalLengthCm'] > 1.9:
s.add(If(clDt == 0,  # Assuming features[4] corresponds to 'Species'
         Not(Or(features[2] < 1, features[2] > 1.9)),True))  # 1 cm ≤ PetalLengthCm ≤ 1.9 cm for 'Iris-setosa'

s.add(If(clDt == 1,  # Assuming features[4] corresponds to 'Species'
         Not(Or(features[2] < 3, features[2] > 5.1)),True))  # 3 cm ≤ PetalLengthCm ≤ 5.1 cm for 'Iris-versicolor'

s.add(If(clDt == 2,  # Assuming features[4] corresponds to 'Species'
         Not(Or(features[2] < 4.5, features[2] > 6.9)),True))    # 4.5 cm ≤ PetalLengthCm ≤ 6.9 cm for 'Iris-virginica'

#constraint for outlier was: row['PetalWidthCm'] < 0.1 or row['PetalWidthCm'] > 3:
s.add(Not(Or(features[3] < 0.1, features[3] > 3)))
# ... for all of these

# disagreement datasets
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
        print(len(dataset))
        # stop after 10000 elements in the dataset
        if len(dataset) >= 900:
            break

print(dataset)
# replace class labels with class names
for row in dataset:
    row[-1] = class_names[row[-1]]


header = list(X.columns) + ["Species"]

# write dataset to csv file
import csv
with open('iris_diff_data2_plausible.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
    # Sepal length constraint: 4 cm ≤ Sepal length ≤ 8 cm

    if row['SepalLengthCm'] < 4 or row['SepalLengthCm'] >8:
        return True
    
    # Sepal width constraint: 2 cm ≤ Sepal width ≤ 4.5 cm
    
    if row['SepalWidthCm'] < 2 or row['SepalWidthCm'] > 4.5:
        return True
    
    # Petal length and species-specific constraints
    if row['Species'] == 'Iris-setosa':
        # Setosa: 1 cm ≤ Petal length ≤ 2 cm
        if row['PetalLengthCm'] < 1 or row['PetalLengthCm'] > 2:
            return True
    else:
        # Versicolor, Virginica: 2 cm ≤ Petal length ≤ 7 cm
        if row['PetalLengthCm'] < 2 or row['PetalLengthCm'] > 7:
            return True
    
    # Petal width constraint: 0 cm ≤ Petal width ≤ 3 cm
    if row['PetalWidthCm'] < 0.1 or row['PetalWidthCm'] > 3:
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


data_generation_time = time.time() - start_time
print(f"Data Generation Time: {data_generation_time:.2f} seconds")

print(f"Memory used: {process.memory_info().rss / 1024**2:.2f} MB")