import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

print(train.shape, test.shape)
# print(train.info())

train_target = train.pop('price')
train = pd.get_dummies(train)
test = pd.get_dummies(test)

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(train, train_target, test_size=0.2, random_state=0)
# print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_tr, y_tr)
pred = rf.predict(X_val)
# print(pred)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_val, pred)**0.5)
# pred = rf.predict(test)
# # print(pred)

# result = pd.DataFrame({
#     'pred': pred
# })

# result.to_csv('result.csv', index=False)
# csv = pd.read_csv('result.csv')
# print(csv)