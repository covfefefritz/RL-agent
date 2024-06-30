class LearningTrader:
    def __init__(self):
        self.current_position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = None

    def perform_action(self, current_data, action, data_handler):
        spread = 0.0002
        current_price = current_data['Close']
        trade = {'action': 'NA', 'price': current_price, 'timestamp': current_data['Gmt time'], 'size': 0}

        if action == 0:  # Buy
            if self.current_position == 0:
                trade = {'action': 'buy', 'price': current_price + spread, 'timestamp': current_data['Gmt time'], 'size': 1}
                data_handler.place_order(trade)
                self.entry_price = trade['price']
                self.current_position = 1
            elif self.current_position == -1:
                trade = {'action': 'close_short', 'price': current_price + spread, 'timestamp': current_data['Gmt time'], 'size': 1}
                data_handler.place_order(trade)
                self.entry_price = None
                self.current_position = 0

        elif action == 1:  # Sell
            if self.current_position == 0:
                trade = {'action': 'sell', 'price': current_price - spread, 'timestamp': current_data['Gmt time'], 'size': 1}
                data_handler.place_order(trade)
                self.entry_price = trade['price']
                self.current_position = -1
            elif self.current_position == 1:
                trade = {'action': 'close_long', 'price': current_price - spread, 'timestamp': current_data['Gmt time'], 'size': 1}
                data_handler.place_order(trade)
                self.entry_price = None
                self.current_position = 0

        elif action == 2:  # Hold
            trade = {'action': 'hold', 'price': current_price, 'timestamp': current_data['Gmt time'], 'size': 0}

        return trade
