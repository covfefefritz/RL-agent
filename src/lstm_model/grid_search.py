import itertools
import numpy as np
import pandas as pd
import requests
from agent import TradingAgent  # Ensure this imports the updated TradingAgent class
import logging
import time
import os
import matplotlib.pyplot as plt
import gc  # Garbage collection module
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a directory to store the results
results_dir = "grid_search_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Parameter ranges
stop_loss_range = [0.97, 0.98, 0.99]
buy_threshold_range = [1.0025, 1.005, 1.0075]
sell_threshold_range = [0.9975, 0.995, 0.9925]
max_positions_range = [1, 2, 3]
prediction_steps_range = [5, 10, 15]

# Grid search
best_performance = -np.inf
best_params = None
overall_results = []

param_combinations = list(itertools.product(stop_loss_range, buy_threshold_range, sell_threshold_range, max_positions_range, prediction_steps_range))

def reset_api_feed():
    reset_url = "http://api:5000/reset_data"
    try:
        response = requests.post(reset_url, timeout=10)
        response.raise_for_status()
        logging.info("API data feed reset.")
    except requests.RequestException as e:
        logging.error(f"Error resetting API data feed: {e}")

for stop_loss, buy_threshold, sell_threshold, max_positions, prediction_steps in param_combinations:
    logging.info(f"Testing parameters: stop_loss={stop_loss}, buy_threshold={buy_threshold}, sell_threshold={sell_threshold}, max_positions={max_positions}, prediction_steps={prediction_steps}")
    
    # Reset API data feed
    reset_api_feed()

    # Initialize the trading agent with the current parameter combination
    agent = TradingAgent(
        model_path='lstm_model_v3_simple.h5',
        scaler_path='scaler_v3_simple.pkl',
        api_url='http://api:5000/get_data',
        stop_loss=stop_loss,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        prediction_steps=prediction_steps
    )

    # Run the trading simulation
    try:
        while True:
            new_data = agent.fetch_new_data()
            if new_data is None:
                break  # Exit the loop if no more data is available
            agent.update_data_and_predict()
            agent.execute_trades()
            time.sleep(0.1)  # Reduce sleep time to speed up the process, but not too much
    except KeyboardInterrupt:
        logging.info("Terminating the agent.")
    except Exception as e:
        logging.error(f"Error during trading simulation: {e}")
        break
    
    cumulative_profit_loss = agent.cumulative_profit_loss
    number_of_trades = agent.number_of_trades
    max_drawdown = agent.max_drawdown

    overall_results.append({
        'stop_loss': stop_loss,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'max_positions': max_positions,
        'prediction_steps': prediction_steps,
        'cumulative_profit_loss': cumulative_profit_loss,
        'number_of_trades': number_of_trades,
        'max_drawdown': max_drawdown
    })

    # Save individual results to a CSV file
    individual_result_file = os.path.join(
        results_dir,
        f"results_stop_loss_{stop_loss}_buy_threshold_{buy_threshold}_sell_threshold_{sell_threshold}_max_positions_{max_positions}_prediction_steps_{prediction_steps}.csv"
    )
    pd.DataFrame(agent.trading_log).to_csv(individual_result_file, index=False)

    # Determine performance based on new criteria
    performance = cumulative_profit_loss - max_drawdown  # You can adjust this formula as needed
    if performance > best_performance:
        best_performance = performance
        best_params = (stop_loss, buy_threshold, sell_threshold, max_positions, prediction_steps)
    
    # Explicitly delete the agent and call garbage collection
    del agent
    gc.collect()

    # Reset TensorFlow state to avoid memory buildup
    tf.keras.backend.clear_session()
    gc.collect()

    # Log memory usage
    logging.info(f"Memory usage after garbage collection: {gc.get_stats()}")

# Save overall results to CSV
overall_results_df = pd.DataFrame(overall_results)
overall_results_file = os.path.join(results_dir, "overall_grid_search_results.csv")
overall_results_df.to_csv(overall_results_file, index=False)

# Print the best parameters and performance
logging.info(f"Best parameters: stop_loss={best_params[0]}, buy_threshold={best_params[1]}, sell_threshold={best_params[2]}, max_positions={best_params[3]}, prediction_steps={best_params[4]}")
logging.info(f"Best performance: {best_performance}")

# Visualization and save the plot
plt.figure(figsize=(12, 8))
plt.plot(overall_results_df.index, overall_results_df['cumulative_profit_loss'], marker='o', linestyle='-')
plt.xlabel('Parameter Combination Index')
plt.ylabel('Cumulative Profit/Loss')
plt.title('Grid Search Results')
plt.xticks(ticks=range(len(param_combinations)), labels=[f'Combo {i}' for i in range(len(param_combinations))], rotation=90)
plt.grid(True)
plot_file = os.path.join(results_dir, "grid_search_results_plot.png")
plt.savefig(plot_file)
plt.show()
