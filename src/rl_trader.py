import logging

class LearningTrader:
    def __init__(self):
        self.max_position_size = 3
        self.current_position = 0  # Net position size
        self.entry_prices = []  # List to track entry prices and sizes
        self.exit_price = None
        self.trades = []

    def reset(self):
        self.current_position = 0
        self.entry_prices = []
        self.exit_price = None
        self.trades = []

    def perform_action(self, current_data, scaled_action, data_handler, instrument):
        trade_action = None  # Default assignment
        trade = {'action': 'hold', 'price': None, 'timestamp': current_data['Gmt time'], 'size': 0, 'success': False}
        new_position_size = self.current_position + scaled_action

        # Reject trades that would exceed max position size
        if abs(new_position_size) > self.max_position_size:
            logging.warning("Trade rejected: exceeds max position size. Current: %d, Attempted: %d", self.current_position, new_position_size)
            return trade

        if scaled_action > 0:  # Buy action
            if self.current_position >= 0:  # Flat or long
                trade_action = 'buy'
            elif self.current_position < 0:  # Short position
                if new_position_size < 0:
                    trade_action = 'reduce_short'
                elif new_position_size == 0:
                    trade_action = 'close_short'
        elif scaled_action < 0:  # Sell action
            if self.current_position <= 0:  # Flat or short
                trade_action = 'sell'
            elif self.current_position > 0:  # Long position
                if new_position_size > 0:
                    trade_action = 'reduce_long'
                elif new_position_size == 0:
                    trade_action = 'close_long'
        
        # Ensure trade_action is not None before proceeding
        if trade_action is None:
            logging.warning("No valid trade action determined, skipping trade.")
            return trade

        trade = {'action': trade_action, 'price': None, 'timestamp': current_data['Gmt time'], 'size': abs(scaled_action)}
        trade['success'] = data_handler.place_order(trade, trade_action, instrument, abs(scaled_action))
        if trade['success']:
            logging.info("%s order placed: %s", trade_action.capitalize(), trade)
            self.current_position = new_position_size
            if trade_action == 'buy' or trade_action == 'sell':
                self.entry_prices.append({'price': trade['price'], 'size': abs(scaled_action)})
            elif trade_action == 'close_long' or trade_action == 'close_short':
                self.entry_prices = self.close_position(abs(scaled_action))

        return trade

    def close_position(self, size):
        remaining_size = size
        updated_entries = []
        for entry in self.entry_prices:
            if remaining_size <= 0:
                updated_entries.append(entry)
                continue
            if entry['size'] > remaining_size:
                entry['size'] -= remaining_size
                remaining_size = 0
                updated_entries.append(entry)
            else:
                remaining_size -= entry['size']
        return updated_entries

    def update_position(self, filled_trade):
        if filled_trade and filled_trade['success']:
            action = filled_trade['action']
            price = filled_trade.get('price', None)
            fee = filled_trade.get('fee', 0)  # Get the fee from the filled trade, default to 0 if not present
            size = filled_trade.get('size', 1)  # Get the size from the filled trade, default to 1 if not present

            if price is None:
                logging.warning("Filled trade has no price: %s", filled_trade)
                return

            if action in ['buy', 'sell', 'close_long', 'close_short', 'reduce_long', 'reduce_short']:
                if action in ['buy', 'sell']:
                    # Apply fee to the entry price
                    adjusted_price = price + (fee*size) if action == 'buy' else price - (fee*size)
                    self.entry_prices.append({'price': adjusted_price, 'size': size})
                elif action in ['close_long', 'close_short']:
                    self.exit_price = price
                    self.entry_prices = self.close_position(size)

                self.trades.append(filled_trade)
                logging.info("Trade filled and position updated: %s", filled_trade)
            else:
                logging.warning("Unknown action type in filled trade: %s", action)
        else:
            logging.warning("No trade was filled to update position.")
