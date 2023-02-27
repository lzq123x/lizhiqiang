import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
import requests
import json
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_epochs = 10


# Set number of days to retrieve from API
num_days = 30

# Set lottery type and API URL
lottery_type = 'bjpk10'
url = f'https://www.1680263.com/api/pks/getPksHistoryList.do?lotCode=10058'

# Set directory to save data
data_dir = './lishiwenjian/'

# Calculate start and end dates
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=num_days)).strftime('%Y-%m-%d')

# Delete today's data
today_file = os.path.join(data_dir, end_date + '.json')
if os.path.isfile(today_file):
    os.remove(today_file)

# Check if data is missing and fetch missing data
full_data = []
for i in range(num_days):
    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
    data_file_name = os.path.join(data_dir, date + '.json')
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

np_new_features = np.array(np_new_features[1:])
if np_pre_draw_codes.shape[0] - np_new_features.shape[0] == 1:
    np_pre_draw_codes = np_pre_draw_codes[:-1]
np_X = np.concatenate((np_pre_draw_codes[:-1], np_new_features[:-1]), axis=1)
np_y = np_pre_draw_codes[1:, 0]

class LotteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

train_dataset = LotteryDataset(np_X[:, :, :, :], np_y)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = LotteryDataset(np_X[-10:, :, :, :], np_y[-10:])
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

# Train CNN model
class LotteryCNN(nn.Module):
    def __init__(self):
        super(LotteryCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=(1,2))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = LotteryCNN()
if use_cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
valid_loss_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    # Train the model
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validate the model
    model.eval()
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        valid_loss += loss.item()

    train_loss_list.append(train_loss / len(train_loader))
    valid_loss_list.append(valid_loss / len(valid_loader))

    # Print progress
    print('[%d] Train loss: %.3f Valid loss: %.3f' %
          (epoch + 1, train_loss_list[-1], valid_loss_list[-1]))

# Predict next draw's first number and print probabilities of top 5 numbers
model.eval()

with torch.no_grad():
    input_data = torch.Tensor(np.array([np_pre_draw_codes[-1]])).unsqueeze(1)
    if use_cuda:
        input_data = input_data.cuda()
    output = model(input_data)
    probas = F.softmax(output, dim=1)[0]
    pred_num_index = np.argsort(probas.cpu().numpy())[-5:][::-1]
    pred_num_prob = probas[pred_num_index]
    for i, (index, p) in enumerate(zip(pred_num_index, pred_num_prob)):
        print("第{}预测数字{}的概率：{:.2f}%".format(i+1, index+1, p*100))

# Improve prediction by checking if predicted number matches recent history
pred_num = num
for i in range(2, 6):
    if np_pre_draw_codes[-i, 0] == pred_num:
        pred_num_index = np.argsort(probas[1:])[-4:][::-1]
        pred_num = pred_num_index[0] + 2
        break

# Print the final prediction
results = []
for i, index in enumerate(np.argsort(probas)[-7:][::-1]):
    prob = probas[index]
    results.append((index+1, prob*100))

print("Final prediction:")
for i, (num, prob) in enumerate(results):
    print("Rank {}: Number {}, Probability {:.2f}%".format(i+1, num, prob))

# Fetch latest draw data from API
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    latest_draw = list(map(int, data['result']['data'][0]['preDrawCode'].split(',')))
else:
    print("无法获取数据。")
    exit()

# Print latest draw
print("最新开奖：", latest_draw)


