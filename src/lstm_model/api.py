from flask import Flask, jsonify
import pandas as pd
import time
from threading import Thread, Event, Lock

app = Flask(__name__)

# Load the dummy data
data = pd.read_csv('../historic_data/unprocessed_data/EURUSD_Candlestick_1_Hour_BID_01.04.2022-13.04.2024.csv')
index = 1
reset_event = Event()
lock = Lock()

def update_index():
    global index
    while not reset_event.is_set():
        with lock:
            if index < len(data):
                index += 1
        time.sleep(0.10)

@app.route('/get_data', methods=['GET'])
def get_data():
    global index
    with lock:
        if index < len(data):
            row = data.iloc[index].to_dict()
            row['index'] = index  # Add the current index to the response
            return jsonify(row)
        else:
            return jsonify({'error': 'No more data'}), 404

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global index
    with lock:
        index = 1
    reset_event.clear()  # Clear the event to allow the index updating to start again
    thread = Thread(target=update_index)
    thread.start()
    return jsonify({'message': 'Data feed reset'}), 200

@app.route('/stop_data', methods=['POST'])
def stop_data():
    reset_event.set()  # Stop the index updating
    return jsonify({'message': 'Data feed stopped'}), 200

if __name__ == '__main__':
    try:
        reset_event.clear()  # Initialize the event
        thread = Thread(target=update_index)
        thread.start()
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        reset_event.set()  # Ensure the event is set to stop the thread on shutdown
