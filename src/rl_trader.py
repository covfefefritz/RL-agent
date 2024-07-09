import logging

class LearningTrader:
    def __init__(self):
        self.current_position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = None
        self.exit_price = None
        self.trades = []

    def perform_action(self, current_data, action, data_handler):
        spread = 0.0002
        current_price = current_data['Close']
        trade = {'action': 'hold', 'price': None, 'timestamp': current_data['Gmt time'], 'size': 0, 'success': False}

        if action == 0:  # Buy
            if self.current_position == 0:
                trade = {'action': 'buy', 'price': None, 'timestamp': current_data['Gmt time'], 'size': 1}
                trade['success'] = data_handler.place_order(trade, 'buy', spread)
                if trade['success']:
                    logging.info("Buy order placed: %s", trade)
            elif self.current_position == -1:
                trade = {'action': 'close_short', 'price': None, 'timestamp': current_data['Gmt time'], 'size': 1}
                trade['success'] = data_handler.place_order(trade, 'close_short', spread)
                if trade['success']:
                    logging.info("Close short order placed: %s", trade)

        elif action == 1:  # Sell
            if self.current_position == 0:
                trade = {'action': 'sell', 'price': None, 'timestamp': current_data['Gmt time'], 'size': 1}
                trade['success'] = data_handler.place_order(trade, 'sell', spread)
                if trade['success']:
                    logging.info("Sell order placed: %s", trade)
            elif self.current_position == 1:
                trade = {'action': 'close_long', 'price': None, 'timestamp': current_data['Gmt time'], 'size': 1}
                trade['success'] = data_handler.place_order(trade, 'close_long', spread)
                if trade['success']:
                    logging.info("Close long order placed: %s", trade)

        return trade


    def update_position(self, filled_trade):
        if filled_trade and filled_trade['success']:
            if filled_trade['action'] == 'buy':
                self.entry_price = filled_trade['price']
                self.current_position = 1
            elif filled_trade['action'] == 'sell':
                self.entry_price = filled_trade['price']
                self.current_position = -1
            elif filled_trade['action'] == 'close_long' or filled_trade['action'] == 'close_short':
                self.exit_price = filled_trade['price']
                self.current_position = 0
            self.trades.append(filled_trade)  # Add trade only if successful
            logging.info("Trade filled and position updated: %s", filled_trade)
        else:
            logging.warning("No trade was filled to update position.")
