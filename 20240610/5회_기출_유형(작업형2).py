import pandas as pd
train = pd.read_csv('./20240611/5th/train.csv')
test = pd.read_csv('./20240611/5th/test.csv')
# print(train)
# print(test)
# print(train.info())
# print(train.unique()) ???
# print(train.isnull().sum())
# print(test.isnull().sum())

train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.info())

train_price = train.pop("price")

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(train, train_price)
pred = rf.predict(test)
print(pred)
result = pd.DataFrame({
    'pred': pred
})

result.to_csv('result.csv', index=False)