import pandas as pd
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# print(train.head())

# print(train.info())

train_id = train.pop('ID')
test_id = test.pop('ID')
train_target = train.pop('Segmentation')
train = pd.get_dummies(train)
test = pd.get_dummies(test)
# from sklearn.model_selection import train_test_split
# X_tr, x_val, y_tr, y_val = train_test_split(train, train_target, test_size=0.2, random_state=0)
# print(test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(train, train_target)
# model = rf.fit(X_tr, y_tr)


# pred = model.predict(x_val)
# print(pred)
# print(y_val)
# from sklearn.metrics import f1_score
# print(f1_score(y_val, pred))
pred = model.predict(test)
print(pred)

pd.DataFrame({
    'ID': test_id,
    'Segmentation': pred
}).to_csv('result.csv', index=False)

df = pd.read_csv('result.csv')
print(df)