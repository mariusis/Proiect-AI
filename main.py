import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



P = Perceptron()

data_train = pd.read_csv("letters/emnist-letters-train.csv", header=None)
letter_a = data_train[data_train.iloc[:, 0] == 1]-1
letter_b = data_train[data_train.iloc[:, 0] == 2]-1
letter_c = data_train[data_train.iloc[:, 0] == 3]-1
letter_d = data_train[data_train.iloc[:, 0] == 4]-1
letter_e = data_train[data_train.iloc[:, 0] == 5]-1
letter_f = data_train[data_train.iloc[:, 0] == 6]-1
letter_g = data_train[data_train.iloc[:, 0] == 7]-1
letter_h = data_train[data_train.iloc[:, 0] == 8]-1
letter_i = data_train[data_train.iloc[:, 0] == 9]-1
letter_j = data_train[data_train.iloc[:, 0] == 10]-1
letter_k = data_train[data_train.iloc[:, 0] == 11]-1
letter_l = data_train[data_train.iloc[:, 0] == 12]-1
letter_m = data_train[data_train.iloc[:, 0] == 13]-1
letter_n = data_train[data_train.iloc[:, 0] == 14]-1
letter_o = data_train[data_train.iloc[:, 0] == 15]-1
letter_p = data_train[data_train.iloc[:, 0] == 16]-1
letter_q = data_train[data_train.iloc[:, 0] == 17]-1
letter_r = data_train[data_train.iloc[:, 0] == 18]-1
letter_s = data_train[data_train.iloc[:, 0] == 19]-1
letter_t = data_train[data_train.iloc[:, 0] == 20]-1
letter_u = data_train[data_train.iloc[:, 0] == 21]-1
letter_v = data_train[data_train.iloc[:, 0] == 22]-1
letter_w = data_train[data_train.iloc[:, 0] == 23]-1
letter_x = data_train[data_train.iloc[:, 0] == 24]-1
letter_y = data_train[data_train.iloc[:, 0] == 25]-1
letter_z = data_train[data_train.iloc[:, 0] == 26]-1

train_a_z = [letter_a, letter_b, letter_c, letter_d, letter_e, letter_f, letter_g,
             letter_h, letter_i, letter_j, letter_k, letter_l, letter_m, letter_n,
             letter_o, letter_p, letter_q, letter_r, letter_s, letter_t, letter_u,
             letter_v, letter_w, letter_x, letter_y, letter_z]

data = pd.concat(train_a_z)

train_data, test_data = train_test_split(data, test_size=0.25, random_state=1, shuffle=True)

x_train = train_data.drop(train_data.columns[0], axis=1).to_numpy()
x_test = test_data.drop(test_data.columns[0], axis=1).to_numpy()
y_train = train_data.iloc[:, 0].to_numpy()
y_test = test_data.iloc[:, 0].to_numpy()

#Normalizare
x_train = x_train/255
x_test = x_test/255

P.fit(x_train, y_train)

prediction = P.predict(x_test)
report = classification_report(prediction, y_test, digits=6, labels=np.unique(prediction))
print(report)

data_test = pd.read_csv('letters/emnist-letters-test.csv')
letter_a = data_train[data_train.iloc[:, 0] == 1]-1
letter_b = data_train[data_train.iloc[:, 0] == 2]-1
letter_c = data_train[data_train.iloc[:, 0] == 3]-1
letter_d = data_train[data_train.iloc[:, 0] == 4]-1
letter_e = data_train[data_train.iloc[:, 0] == 5]-1
letter_f = data_train[data_train.iloc[:, 0] == 6]-1
letter_g = data_train[data_train.iloc[:, 0] == 7]-1
letter_h = data_train[data_train.iloc[:, 0] == 8]-1
letter_i = data_train[data_train.iloc[:, 0] == 9]-1
letter_j = data_train[data_train.iloc[:, 0] == 10]-1
letter_k = data_train[data_train.iloc[:, 0] == 11]-1
letter_l = data_train[data_train.iloc[:, 0] == 12]-1
letter_m = data_train[data_train.iloc[:, 0] == 13]-1
letter_n = data_train[data_train.iloc[:, 0] == 14]-1
letter_o = data_train[data_train.iloc[:, 0] == 15]-1
letter_p = data_train[data_train.iloc[:, 0] == 16]-1
letter_q = data_train[data_train.iloc[:, 0] == 17]-1
letter_r = data_train[data_train.iloc[:, 0] == 18]-1
letter_s = data_train[data_train.iloc[:, 0] == 19]-1
letter_t = data_train[data_train.iloc[:, 0] == 20]-1
letter_u = data_train[data_train.iloc[:, 0] == 21]-1
letter_v = data_train[data_train.iloc[:, 0] == 22]-1
letter_w = data_train[data_train.iloc[:, 0] == 23]-1
letter_x = data_train[data_train.iloc[:, 0] == 24]-1
letter_y = data_train[data_train.iloc[:, 0] == 25]-1
letter_z = data_train[data_train.iloc[:, 0] == 26]-1

test_a_z = [letter_a, letter_b, letter_c, letter_d, letter_e, letter_f, letter_g,
             letter_h, letter_i, letter_j, letter_k, letter_l, letter_m, letter_n,
             letter_o, letter_p, letter_q, letter_r, letter_s, letter_t, letter_u,
             letter_v, letter_w, letter_x, letter_y, letter_z]

validate_data = pd.concat(test_a_z).sample(frac=1)

x_validate = validate_data.drop(validate_data.columns[0], axis=1).to_numpy()
y_validate = validate_data.iloc[:, 0].to_numpy()

prediction = P.predict(x_validate)

report = classification_report(prediction, y_validate, digits=6, labels=np.unique(prediction))
print(report)
