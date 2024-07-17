import pandas as pd
import plotly.graph_objects as go

# Constants
INITIAL_EQUITY = 1000
PIP_VALUE = 1  # $1 per pip
FEE_PER_CONTRACT = 3  # 3 pips per contract

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

# Initialize equity tracking
equity = INITIAL_EQUITY
equity_curve = []
open_positions = []

# Function to calculate equity change for a trade
def calculate_equity_change(action, price, size, equity, is_open):
    global FEE_PER_CONTRACT, PIP_VALUE
    fee = FEE_PER_CONTRACT * size * PIP_VALUE
    equity_change = 0

    if is_open:
        # Deduct the fee immediately when the position is opened
        equity_change = -fee
    else:
        # Calculate the profit/loss and deduct the fee when the position is closed or reduced
        if action in ['close_long', 'reduce_long']:
            direction = 1  # Long position
        else:
            direction = -1  # Short position

        if open_positions:
            entry_price = open_positions.pop(0)['price']  # Assuming FIFO for positions
            price_change = (price - entry_price) * direction * size * PIP_VALUE
            equity_change = price_change - fee

    return equity + equity_change

# Process trades in chronological order
for idx, trade in trade_log.iterrows():
    if trade['action'] in ['buy', 'sell']:
        # Open a new position
        open_positions.append(trade)
        equity = calculate_equity_change(trade['action'], trade['price'], trade['size'], equity, True)
    elif trade['action'] in ['close_long', 'close_short', 'reduce_long', 'reduce_short']:
        # Close or reduce a position
        if open_positions:
            equity = calculate_equity_change(trade['action'], trade['price'], trade['size'], equity, False)
    
    equity_curve.append((trade['timestamp'], equity))

# Consider all positions closed at the end of the period
for position in open_positions:
    # Close all remaining open positions
    if position['action'] == 'buy':
        close_action = 'close_long'
    else:
        close_action = 'close_short'

    # Use the last available closing price for the remaining positions
    last_price = filtered_price_data.iloc[-1]['Close']
    equity = calculate_equity_change(close_action, last_price, position['size'], equity, False)
    equity_curve.append((position['timestamp'], equity))

# Create equity curve DataFrame
equity_curve_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])

# Plot the equity curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_curve_df['timestamp'], y=equity_curve_df['equity'], mode='lines', name='Equity'))

# Add candlestick data
fig.add_trace(go.Candlestick(
    x=filtered_price_data['timestamp'],
    open=filtered_price_data['Open'],
    high=filtered_price_data['High'],
    low=filtered_price_data['Low'],
    close=filtered_price_data['Close'],
    name='Price'
))

# Function to stack markers based on size
def add_stacked_markers(fig, actions, action_type, color, symbol):
    for _, row in actions.iterrows():
        for i in range(row['size']):
            y_offset = 0.0005 * i  # Adjust the offset value as needed
            fig.add_trace(go.Scatter(
                x=[row['timestamp']], 
                y=[row['price'] + y_offset if action_type in ['buy', 'reduce_short', 'close_short'] else row['price'] - y_offset], 
                mode='markers', 
                name=action_type.capitalize(),
                marker=dict(color=color, symbol=symbol, size=10),
                text=f"{action_type.capitalize()} @ {row['price']}",
                hoverinfo='text'
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
    title='Forex Price Data with Trade Actions and Equity Curve',
    xaxis_title='Timestamp',
    yaxis_title='Price/Equity',
    legend_title='Legend',
    template='plotly_dark'
)

# Save the plot to a file
fig.write_html('forex_price_with_trade_actions_and_equity.html')

# Alternatively, show the plot in a browser
# fig.show()
