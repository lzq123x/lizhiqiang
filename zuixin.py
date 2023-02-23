import requests
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import os.path
import time
import configparser


def get_data():
    # read configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    url = config['default']['url']
    num_days = int(config['default']['num_days'])
    history_dir = config['default']['history_dir']
    prediction_dir = config['default']['prediction_dir']
    model_path = config['default']['model_path']

    # get date range to retrieve data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=num_days)).strftime("%Y-%m-%d")

    # check if data has been previously retrieved
    history_file = os.path.join(history_dir, f"data_{start_date}_{end_date}.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            data = json.load(f)
    else:
        # retrieve data from API
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299 '
        }
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # save retrieved data to history file
        with open(history_file, 'w') as f:
            json.dump(data, f)

    # process data
    values = np.array(data['data'])
    dates = np.array(data['labels'])
    dates = np.array([datetime.strptime(d, '%Y-%m-%d') for d in dates])

    # train model if not already trained
    if os.path.exists(model_path):
        # load model
        clf = MLPClassifier()
        clf = clf.partial_fit([values[-1]], [predict_class(values[-1])], classes=[0, 1])
    else:
        # train model
        parameters = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
        clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
        clf.fit(values, predict_class(values))
        clf = clf.best_estimator_
        clf.partial_fit([values[-1]], [predict_class(values[-1])], classes=[0, 1])

        # save model to file
        if not os.path.exists(model_file):
            clf.fit(X_train, y_train)
            with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        # load model from file
        with open(model_file, 'rb') as f:
            clf = pickle.load(f)

        # predict
        X_today = np.array(today_data).reshape(1, -1)
        result = clf.predict(X_today)

        # output result
        if result == 1:
            print(today_str, "今日中奖！")
        else:
            print(today_str, "今日无中奖")

