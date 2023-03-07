import numpy as np
from collections import Counter
from random import sample

mushrooms = np.loadtxt('dane/mushroom-modified-filled.txt', dtype=str)
mushrooms_types = np.loadtxt('dane/mushroom-modified-filled-type.txt', dtype=str)

# Zad3
# a)
decision_classes = np.unique(mushrooms[:, 22])
print("\nAvailable decision classes: {}".format(decision_classes))
# b)
counts = Counter(mushrooms[:, 22])
print("\nNumber of objects in decision classes")
for key, value in counts.items():
    print(key, value)

# c) Only Symbolic Attributes in Selected Decision System
# d), e)
diff_value_count = {}
diff_value = {}
common_value = {}
attributes = mushrooms_types[:, 0]
for i in range(len(attributes)):
    diff_value_count[attributes[i]] = len(set(mushrooms[:, i]))
    diff_value[attributes[i]] = set(mushrooms[:, i])
    counts = Counter(mushrooms[:, i])
    common_value[attributes[i]] = counts.most_common()[0][0]

print("\nNumber of unique attribute values")
for key, value in diff_value_count.items():
    print(key, value)
print("\nUnique attribute values")
for key, value in diff_value.items():
    print(key, value)
print("\nCommon value")
for key, value in common_value.items():
    print(key, value)
# f) Only Symbolic Attributes in Selected Decision System

# Zad4
# a)
row = mushrooms.shape[0]
column = mushrooms.shape[1] - 1
missing_values_count = int(row * column * 0.1)

rand_row = sample(range(0, 8124), 8124)
rand_column = sample(range(0, 22), 22)
mush = mushrooms.copy()
common = list(common_value.values())

for i in range(missing_values_count):
    mush[rand_row[i % row]][rand_column[i % column]] = '?'

assert np.count_nonzero(mush == '?') == missing_values_count

print("\nAfter generated ten per cent of missing values in selected decision system")
print(mush)

for row_number in range(row):
    for value in range(column):
        if mush[row_number][value] == '?':
            mush[row_number][value] = common[value]

assert np.count_nonzero(mush == '?') == 0
print("\nAfter complete the missing values with most common values in selected decision system")
print(mush)

# b) c) Only Symbolic Attributes in Selected Decision System
# d)
churn = np.loadtxt('dane/Churn_Modelling.csv', delimiter=",", dtype=str)
print("\nLoaded csv file")
print(churn)
geography_index = np.where(churn[0] == "Geography")[0][0]
geography = churn[1:, geography_index:geography_index + 1]
counter = Counter(geography[:, 0])
dummy = np.zeros((churn.shape[0], 1), dtype=churn.dtype)
dummy_columns = np.zeros((churn.shape[0], 0), dtype=churn.dtype)
for j in counter.keys():
    dummy[0] = j
    for i in range(1, geography.size+1):
        if geography[i-1] == j:
            dummy[i] = 1
        else:
            dummy[i] = 0
    dummy_columns = np.append(dummy_columns, dummy, axis=1)
    dummy = np.zeros((churn.shape[0], 1), dtype=churn.dtype)

print("\nDummy variables - First step")
print(dummy_columns)
churn = np.append(churn, dummy_columns, axis=1)
to_delete = sorted(counter.most_common(), reverse=True)[0][0]
delete_column_index = np.where(churn[0] == to_delete)[0][0]
churn = np.delete(churn, delete_column_index, axis=1)
print("\nAdded Dummy variables and removed {} variable - Last step".format(to_delete))
print(churn)
