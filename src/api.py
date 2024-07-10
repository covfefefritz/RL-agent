from flask import Flask, jsonify, request
import pandas as pd
import random
import threading

app = Flask(__name__)

# Load the dummy data
data = pd.read_csv('historic_data/EURUSD-train.csv')
index = 0
lock = threading.Lock()

@app.route('/get_data', methods=['GET'])
def get_data():
    global index
    with lock:
        if index < len(data):
            row = data.iloc[index].to_dict()
            row['index'] = index  # Add the current index to the response
            index += 1
            return jsonify(row)
        else:
            return jsonify({'error': 'No more data'}), 404

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global index
    start = 1000
    end = len(data) - 2000
    with lock:
        index = random.randint(start, end)
    return jsonify({'message': 'Data feed reset',
                    'index': index}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
