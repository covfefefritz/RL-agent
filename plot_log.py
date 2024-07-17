import pandas as pd
import plotly.graph_objects as go

# Load price data
price_data = pd.read_csv('./src/historic_data/EURUSD-train.csv', parse_dates=['Gmt time'], dayfirst=True)
price_data.rename(columns={'Gmt time': 'timestamp'}, inplace=True)

# Load trade log data
trade_log = pd.read_csv('./src/trade_log3.csv')

# Convert timestamps, filling any date-only entries with a default time
trade_log['timestamp'] = pd.to_datetime(trade_log['timestamp'], errors='coerce')
trade_log['timestamp'] = trade_log['timestamp'].fillna(pd.to_datetime(trade_log['timestamp'].dt.strftime('%Y-%m-%d') + ' 00:00:00'))

# Ensure all timestamps are in datetime format
trade_log['timestamp'] = pd.to_datetime(trade_log['timestamp'])

# Determine the range of timestamps in the trade log
start_time = trade_log['timestamp'].min()
end_time = trade_log['timestamp'].max()

# Filter the price data to this range
filtered_price_data = price_data[(price_data['timestamp'] >= start_time) & (price_data['timestamp'] <= end_time)]

# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=filtered_price_data['timestamp'],
    open=filtered_price_data['Open'],
    high=filtered_price_data['High'],
    low=filtered_price_data['Low'],
    close=filtered_price_data['Close'],
    name='Price',
    showlegend=False  # Hide from legend
)])

# Function to stack markers based on size
def add_stacked_markers(fig, actions, action_type, color, symbol):
    for _, row in actions.iterrows():
        for i in range(row['size']):
            y_offset = 0.0005 * i  # Adjust the offset value as needed
            fig.add_trace(go.Scatter(
                x=[row['timestamp']], 
                y=[row['price'] + y_offset if action_type in ['buy', 'reduce_long', 'close_long'] else row['price'] - y_offset], 
                mode='markers', 
                name=action_type.capitalize(),
                marker=dict(color=color, symbol=symbol, size=10),
                text=f"{action_type.capitalize()} @ {row['price']}",
                hoverinfo='text',
                showlegend=(i == 0)  # Show legend entry only for the first marker of each type
            ))

# Define action types and their corresponding colors and symbols
action_types = {
    'buy': ('blue', 'triangle-up'),
    'reduce_short': ('blue', 'triangle-up'),
    'close_short': ('blue', 'x'),
    'sell': ('red', 'triangle-down'),
    'reduce_long': ('red', 'triangle-down'),
    'close_long': ('red', 'x')
}

# Add markers for each action type
for action_type, (color, symbol) in action_types.items():
    actions = trade_log[trade_log['action'] == action_type]
    add_stacked_markers(fig, actions, action_type, color, symbol)

# Update layout
fig.update_layout(
    title='Forex Price Data with Trade Actions',
    xaxis_title='Timestamp',
    yaxis_title='Price',
    template='plotly_dark'
)

# Save the plot to a file
fig.write_html('forex_price_with_trade_actions.html')

# Alternatively, show the plot in a browser
# fig.show()
