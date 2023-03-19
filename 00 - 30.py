import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras import mixed_precision
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.WARNING)

def fetch_data(lot_code, date, data_dir):
    url = f'https://www.1680263.com/api/pks/getPksHistoryList.do?lotCode={lot_code}&date={date}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        full_data = data['result']['data']
        with open(data_dir + date + '.json', 'w') as outfile:
            json.dump(full_data, outfile)
        return full_data
    else:
        print("无法获取数据。")
        exit()
columns = ['period'] + [f'num_{i}' for i in range(1, 36)]
data = pd.DataFrame(columns=columns)

def generate_date_list(start_date, end_date):
    date_list = []
    while start_date < end_date:
        date_list.append(start_date.strftime('%Y-%m-%d'))
        start_date += timedelta(days=1)
    return date_list

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 8)
date_list = generate_date_list(start_date, end_date)

def process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    data = pd.DataFrame.from_records(data_list)
    
    # 修改列名，以匹配原始列名
    data = data.rename(columns={"preDrawCode": "winning_numbers"})

    X = data.drop(["winning_numbers"], axis=1).values
    y = data["winning_numbers"].values

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    return X, y

def load_data(data_dir, date):
    with open(data_dir + date + '.json') as json_file:
        full_data = json.load(json_file)
    return full_data

def delete_today_data(today_file):
    if os.path.exists(today_file):
        os.remove(today_file)
    else:
        print("今天的 JSON 文件不存在，不需要删除。")
    X = data.drop(["winning_numbers"], axis=1).values
    y = data["winning_numbers"].values

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))


def get_dates(num_days):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=num_days)).strftime('%Y-%m-%d')
    return start_date, end_date

def check_and_fetch_data(lot_code, num_days, data_dir):
    full_data = []
    for i in range(num_days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        data_file_name = data_dir + date + '.json'
        if not os.path.isfile(data_file_name):
            full_data += fetch_data(lot_code, date, data_dir)
        else:
            full_data += load_data(data_dir, date)

    # Fetch the latest data
    today = datetime.now().strftime('%Y-%m-%d')
    fetch_data(lot_code, today, data_dir)

    return full_data

def preprocess_data(full_data):
    pre_draw_code_list = []
    pre_draw_issue_list = []
    for item in full_data:
        pre_draw_code_list.insert(0, item['preDrawCode'])
        pre_draw_issue_list.insert(0, item['preDrawIssue'])

    np_pre_draw_codes = np.array([list(map(int, x.split(','))) for x in pre_draw_code_list])

    # Calculate hot and cold numbers
    num_draws = np_pre_draw_codes.shape[0]
    num_numbers = 10
    number_counts = {i+1: 0 for i in range(num_numbers)}
    for row in np_pre_draw_codes:
        for num in row:
            number_counts[num] += 1
    hot_numbers = np.zeros(num_numbers)
    cold_numbers = np.zeros(num_numbers)

    for i in range(num_numbers):
        hot_numbers[i] = number_counts[i+1] / num_draws
        cold_numbers[i] = 1 - hot_numbers[i]

    hot_numbers_sorted = np.argsort(hot_numbers)[::-1]
    cold_numbers_sorted = np.argsort(cold_numbers)[::-1]

    for i in range(num_numbers):
        hot_numbers[i] = hot_numbers_sorted[i]
        cold_numbers[i] = cold_numbers_sorted[i]

    hot_numbers = hot_numbers.astype(int)
    cold_numbers = cold_numbers.astype(int)

    # Rest of the code
    position_distribution = np.zeros((10, 10))
    for row in np_pre_draw_codes:
        for i, num in enumerate(row):
            position_distribution[num-1][i] += 1
    position_distribution_normalized = position_distribution / np.sum(position_distribution, axis=1, keepdims=True)
    features = []
    latest_draw_champion = np_pre_draw_codes[0][0]

    for i in range(np_pre_draw_codes.shape[0] - 1):
        row = np_pre_draw_codes[i]
        prev_row = np_pre_draw_codes[i - 1]

        latest_draw_champion_history = np.sum(prev_row == latest_draw_champion)

        size_feature = 1 if row[0] > 5 else 0

        parity_feature = row[0] % 2

        champion_position_feature = np.where(prev_row == row[0])[0][0] if row[0] in prev_row else -1

        avg_feature = np.mean(row)

        latest_three_draws = np_pre_draw_codes[i:i+3]

        latest_three_draw_counts = np.zeros((10,))
        for draw in latest_three_draws:
            for num in draw:
                latest_three_draw_counts[num-1] += 1

        features.append([latest_draw_champion_history, size_feature, parity_feature, champion_position_feature, avg_feature] + list(position_distribution_normalized[row[0] - 1]) + list(latest_three_draw_counts))

    np_X = np.array(features)
    np_y = np_pre_draw_codes[1:, 0] - 1
    np_y_one_hot = tf.keras.utils.to_categorical(np_y, num_classes=10)

    scaler = MinMaxScaler()
    np_X_scaled = scaler.fit_transform(np_X)

    k_best = SelectKBest(f_classif, k=4)
    np_X_scaled = k_best.fit_transform(np_X_scaled, np_y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(np_X_scaled, np_y_one_hot, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, np_X_scaled, np_y, np_y_one_hot

def build_and_train_model(X_train, X_test, y_train, y_test):
    # 启用混合精度训练
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Build the model
    input_layer = Input(shape=(X_train.shape[1], 1))
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
    max_pool1 = MaxPooling1D(pool_size=1)(conv1)
    dropout1 = Dropout(0.2)(max_pool1)
    conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(dropout1)
    max_pool2 = MaxPooling1D(pool_size=1)(conv2)
    dropout2 = Dropout(0.2)(max_pool2)
    conv3 = Conv1D(filters=128, kernel_size=1, activation='relu')(dropout2)
    max_pool3 = MaxPooling1D(pool_size=1)(conv3)
    dropout3 = Dropout(0.2)(max_pool3)
    lstm = LSTM(64, activation='relu')(dropout3)
    dropout4 = Dropout(0.2)(lstm)

    output_layer = Dense(10, activation='softmax')(dropout4)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                        epochs=50, batch_size=128, verbose=0,
                        validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
                        callbacks=[early_stopping, model_checkpoint])

    return model, history

def main():
    lot_code = 10012
    num_days = 1
    data_dir = './lishiwenjian/'

    delete_today_data(data_dir + datetime.now().strftime('%Y-%m-%d') + '.json')
    start_date, end_date = get_dates(num_days)
    date_list = generate_date_list(start_date, end_date)

    # 计算每个号码的出现次数
    for date in date_list:
        full_data = fetch_data(lot_code, date, data_dir)
        for record in full_data:
            nums = record['preDrawCode'].split(',')
            for num in nums:
                data.loc[record['preDrawIssue'], f"num_{num}"] = 1

    # 计算最近 5 期、10 期和 20 期内的平均出现次数
    for i in range(1, 36):
        data[f"rolling_mean_5_{i}"] = data[f"num_{i}"].rolling(window=5).mean()
        data[f"rolling_mean_10_{i}"] = data[f"num_{i}"].rolling(window=10).mean()
        data[f"rolling_mean_20_{i}"] = data[f"num_{i}"].rolling(window=20).mean()

    data.dropna(inplace=True)  # 删除包含 NaN 的行

    for date in date_list:
        full_data = fetch_data(lot_code, date, data_dir)
        # 对full_data进行预处理和分析
    
    # 交叉验证部分
    kf = StratifiedKFold(n_splits=5)

    accuracy_scores = []

    for train_index, test_index in kf.split(np_X_scaled, np_y):
        X_train, X_test = np_X_scaled[train_index], np_X_scaled[test_index]
        y_train, y_test = np_y_one_hot[train_index], np_y_one_hot[test_index]

        model, _ = build_and_train_model(X_train, X_test, y_train, y_test)
        y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(np_y[test_index], y_pred)
        accuracy_scores.append(accuracy)

    print("交叉验证准确性分数：", accuracy_scores)
    print("平均准确性分数：", np.mean(accuracy_scores))
    
    # 使用训练数据重新训练模型
    best_model, _ = build_and_train_model(X_train, X_train, y_train, y_train)

    # 预测下一期冠军号码
    last_row = np_X_scaled[-1].reshape(1, -1)
    next_pred = best_model.predict(last_row.reshape(1, last_row.shape[1], 1))
    next_pred_probabilities = next_pred[0]
    pred_num_index = np.argsort(next_pred_probabilities)[-10:][::-1]
    pred_num_prob = next_pred_probabilities[pred_num_index]

    for i, (index, p) in enumerate(zip(pred_num_index, pred_num_prob)):
        print("第{}预测数字{}的概率：{:.2f}%".format(i+1, index+1, p*100))

if __name__ == '__main__':
    main()

    # 从最新的 JSON 文件中读取并打印最新的开奖号码
    data_dir = './lishiwenjian/'
    today_data = load_data(data_dir, datetime.now().strftime('%Y-%m-%d'))
    latest_draw = today_data[0]
    print(f"最新开奖（期号：{latest_draw['preDrawIssue']}）:")
    print(f"{latest_draw['preDrawCode']}")
