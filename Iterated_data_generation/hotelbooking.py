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

data = pd.read_csv("D:\Thesis\Git\ml-diff\constraints\hotel_booking.csv")
# label_encoder_month = LabelEncoder()
# data["arrival_date_month_encoded"] = label_encoder_month.fit_transform(data["arrival_date_month"])
X = data[["lead_time", "arrival_date_week_number", "arrival_date_day_of_month","adults"]]  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["arrival_date_month"]) #Target
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
    row[-1] = class_names[row[-1]]


header = list(X.columns) + ["arrival_date_month"]

# write dataset to csv file
import csv
with open('hotelbooking3.csv', mode='w') as file:
    writer = csv.writer(file)
    # add headers
    writer.writerow(header)
    writer.writerows(dataset)



def is_outlier(row):
# BookingConstraint: Booking in July => 26 ≤ Week number ≤ 32
    if row['arrival_date_month'] == 'July':
        if row['arrival_date_week_number'] < 26 or row['arrival_date_week_number'] > 32:
            return True

    # BookingConstraint: Booking in August => 31 ≤ Week number ≤ 36
    if row['arrival_date_month'] == 'August':
        if row['arrival_date_week_number'] < 31 or row['arrival_date_week_number'] > 36:
            return True

    # BookingConstraint: Booking in May => 18 ≤ Week number ≤ 23
    if row['arrival_date_month'] == 'May':
        if row['arrival_date_week_number'] < 18 or row['arrival_date_week_number'] > 23:
            return True

    # BookingConstraint: Booking in October => 40 ≤ Week number ≤ 45
    if row['arrival_date_month'] == 'October':
        if row['arrival_date_week_number'] < 40 or row['arrival_date_week_number'] > 45:
            return True

    # BookingConstraint: Booking in April => 13 ≤ Week number ≤ 18
    if row['arrival_date_month'] == 'April':
        if row['arrival_date_week_number'] < 13 or row['arrival_date_week_number'] > 18:
            return True

    # Day constraint: April arrival date cant be 31
    if row['arrival_date_month'] == 'April':
        if row['arrival_date_day_of_month'] == 31:
            return True
        
    # Day constraint: 1 < arrival_date_day_of_month < 31
    if row['arrival_date_day_of_month'] < 1 or row['arrival_date_day_of_month'] > 31:
            return -1
            
    # Lead time Constraint: 0 ≤ Lead time ≤ 750
    if row['lead_time'] < 0 or row['lead_time'] > 750:
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