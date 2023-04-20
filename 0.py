# Import necessary libraries
import requests
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import os.path
import time
import shutil

# Set number of days to retrieve from API
num_days = 30

# Set lottery type and API URL
lottery_type = 'bjpk10'
url = f'https://api.api68.com/pks/getPksHistoryList.do?lotCode=10057'

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

# Print shapes of np_pre_draw_codes and np_new_features
print(np_pre_draw_codes.shape, np.array(np_new_features).shape)

np_new_features = np.array(np_new_features[1:])
if np_pre_draw_codes.shape[0] - np_new_features.shape[0] == 1:
    np_pre_draw_codes = np_pre_draw_codes[:-1]
np_X = np.concatenate((np_pre_draw_codes[:-1], np_new_features[:-1]), axis=1)
np_y = np_pre_draw_codes[1:, 0]

# Train MLPClassifier model using GPU
parameters = {'hidden_layer_sizes': [(100,), (200,), (300,)], 'solver': ['adam'], 'batch_size': [32, 64, 128], 'learning_rate_init': [0.01, 0.001]}
cv = GridSearchCV(MLPClassifier(max_iter=500), parameters, cv=5, n_jobs=-1, verbose=1)
start_time = time.time()
cv.fit(np_X, np_y)
end_time = time.time()

# Predict next draw's first number and print probabilities of top 5 numbers
next_draw = np.concatenate((np_pre_draw_codes[-1], np_new_features[-1]), axis=0).reshape(1, -1)
prediction = cv.predict(next_draw)
probas = cv.predict_proba(next_draw)[0]
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
    

predictions = list(zip(pred_num_index, pred_num_prob))

print("预测结果：")
# Print the latest draw code
print("最新开奖号码：", pre_draw_code_list[-1])
