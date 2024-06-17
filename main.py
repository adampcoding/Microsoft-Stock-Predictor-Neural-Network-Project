#data collection
import pandas as pd

data = pd.read_csv("/Users/adamslaboratory/Downloads/Microsoft Dataset.csv")
print(data.head())

#choose applicable columns of data
x = data[["Open", "High", "Low", "Volume"]]
y = data[["Adj Close"]]

# split data into: training, test, cross-validation set
from sklearn.model_selection import train_test_split

X_train, x_temp, Y_train, y_temp = train_test_split(x,y,test_size=0.40,random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp,y_temp,test_size=0.50,random_state=1)

# scale features
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(X_train)
x_train_scaled = scaler.transform(X_train)
x_cv_scaled = scaler.transform(x_cv)
x_test_scaled = scaler.transform(x_test)


#Build model, compile, train
import tensorflow as tf
import keras
from keras import Sequential

model = Sequential([
                keras.layers.Dense(units=100, activation="relu",kernel_regularizer=keras.regularizers.l2(0.0001),input_shape=(x_train_scaled.shape[1],)),
                keras.layers.Dense(units=50, activation="relu",kernel_regularizer=keras.regularizers.l2(0.0001)),
                keras.layers.Dense(units=1, activation="linear",kernel_regularizer=keras.regularizers.l2(0.0001))])


model.compile(loss="mean_squared_error", optimizer = "adam")

model.fit(x_train_scaled,Y_train,epochs=100,batch_size=32)

predictions = model.predict(x_train_scaled)

#Model evaluation: loss for training/cross-validation/test sets
train_loss = model.evaluate(x_train_scaled,Y_train)
cv_loss = model.evaluate(x_cv_scaled,y_cv)
test_loss = model.evaluate(x_test_scaled, y_test)

print(train_loss)
print(cv_loss)
print(test_loss)

#Model Evaluation: R^2 score
from sklearn.metrics import r2_score
test_r2 = r2_score(y_test, model.predict(x_test_scaled))
print(test_r2)


#Results:
#Training set loss: 4.73273229598999
#Cross-validation loss: 5.308679103851318
#Test set loss: 4.526670932769775
#R^2 score: 0.9994108080863953