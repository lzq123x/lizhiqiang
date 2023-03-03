import requests
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os.path
import time
import shutil

# Set number of days to retrieve from API
num_days = 180

# Set lottery type and API URL
lottery_type = 'bjpk10'
url = f'https://www.1680263.com/api/pks/getPksHistoryList.do?lotCode=10058'

# Set directory to save data
data_dir = './lishiwenjian/'

# Calculate start and end dates
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=num_days)).strftime('%Y-%m-%d')

# Delete today's data
today_file = data_dir + end_date + '.json'
if os.path.exists(today_file):
    os.remove(today_file)
else:
    print("今天的JSON文件不存在，不需要删除。")

# Check if data is missing and fetch missing data
full_data = []
for i in range(num_days):
    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
    data_file_name = data_dir + date + '.json'
    if not os.path.isfile(data_file_name):
        response = requests.get(url + f'&date={date}')
        if response.status_code == 200:
            data = response.json()
            full_data += data['result']['data']
            with open(data_file_name, 'w') as outfile:
                json.dump(data['result']['data'], outfile)
        else:
            print("无法获取数据。")
            exit()
    else:
        with open(data_file_name) as json_file:
            full_data += json.load(json_file)

# Extract past draw codes and issue numbers
pre_draw_code_list = []
pre_draw_issue_list = []
for item in reversed(full_data):
    pre_draw_code_list.append(item['preDrawCode'])
    pre_draw_issue_list.append(item['preDrawIssue'])

# Convert codes to numpy array for processing
np_pre_draw_codes = np.array([list(map(int, x.split(','))) for x in pre_draw_code_list])

# Extract additional features
np_new_features = []
for i in range(np_pre_draw_codes.shape[0]):
    row = np_pre_draw_codes[i]
    parity = np.sum(row % 2 == 0)
    size = np.sum(row > 5)
    s = np.sum(row)
    span = np.max(row) - np.min(row)
    np_new_features.append([parity, size, s, span])
    for j in range(len(row)):
        np_new_features[-1].append(row[j])
        if j > 0:
            np_new_features[-1].append(row[j] - row[j-1])
        else:
            np_new_features[-1].append(row[j])

# Print shapes of np_pre_draw_codes and np_new_features
print(np_pre_draw_codes.shape, np.array(np_new_features).shape)

np_new_features = np.array(np_new_features[1:])
if np_pre_draw_codes.shape[0] - np_new_features.shape[0] == 1:
    np_pre_draw_codes = np_pre_draw_codes[:-1]
np_X = np.concatenate((np_pre_draw_codes[:-1], np_new_features[:-1]), axis=1)
np_y = np_pre_draw_codes[1:, 0]  # 只取下一期的第一个数字

# Train models and use voting to improve prediction
parameters_rfc = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 4, 8]}
rfc = GridSearchCV(RandomForestClassifier(), parameters_rfc, cv=5, n_jobs=-1, verbose=1)
rfc.fit(np_X, np_y)

parameters_mlp = {'hidden_layer_sizes': [(32,), (64,), (128,), (256,)], 'activation': ['relu', 'tanh', 'logistic'], 'solver': ['adam', 'sgd']}
mlp = GridSearchCV(MLPClassifier(), parameters_mlp, cv=5, n_jobs=-1, verbose=1)
mlp.fit(np_X, np_y)

voting_clf = VotingClassifier([('rfc', rfc.best_estimator_), ('mlp', mlp.best_estimator_)], voting='soft')
voting_clf.fit(np_X, np_y)

# Cross-validate the voting classifier
scores = cross_val_score(voting_clf, np_X, np_y, cv=5)
print("Cross-validation scores: ", scores)

# Predict next draw's first number and print probabilities of top 5 numbers
next_draw = np.concatenate((np_pre_draw_codes[-1], np_new_features[-1]), axis=0).reshape(1, -1)
prediction = voting_clf.predict(next_draw)
probas = voting_clf.predict_proba(next_draw)[0]
pred_num_index = np.argsort(probas)[-5:][::-1]
pred_num_prob = probas[pred_num_index]
for i, (index, p) in enumerate(zip(pred_num_index, pred_num_prob)):
    print("第{}预测数字{}的概率：{:.2f}%".format(i+1, index+1, p*100))

# Improve prediction by checking if predicted number matches recent history
pred_num = int(np.argmax(probas)) + 1
for i in range(2, 6):
    if np_pre_draw_codes[-i, 0] == pred_num:
        pred_num_index = np.argsort(probas[1:])[-4:][::-1]
        pred_num = pred_num_index[0] + 2
        break

results = []
for i, (index, p) in enumerate(zip(pred_num_index, probas[pred_num_index])):
    results.append((index+1, p*100))

print("Prediction for issue {}:".format(pre_draw_issue_list[-1]+1))
for i, (num, prob) in enumerate(results):
    print("Rank {}: Number {}, Probability {:.2f}%".format(i+1, num, prob))

# Check if predicted number matches recent history
pred_num = num
for i in range(2, 6):
    if np_pre_draw_codes[-i, 0] == pred_num:
        pred_num_index = np.argsort(probas[1:])[-4:][::-1]
        pred_num = pred_num_index[0] + 2
        break

# Print the final prediction
results = []
for i, index in enumerate(np.argsort(probas)[-9:][::-1]):
    prob = probas[index]
    results.append((index+1, prob*100))

print("Final prediction:")
for i, (num, prob) in enumerate(results):
    print("Rank {}: Number

