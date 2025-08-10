"""
This is main module for strategy backtesting
"""

import numpy as np
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os

from config.config import BACKTESTING_CONFIG
from database.data_service import DataService
from metrics.metric import Metric
from utils import (
    get_expired_dates,
    from_cash_to_tradeable_contracts,
)

FEE_PER_CONTRACT = Decimal(BACKTESTING_CONFIG["fee"]) * Decimal('100')


class Backtesting:
    """
    Backtesting main class
    """

    def __init__(
        self,
        capital: Decimal,
        printable=True,
    ):
        """
        Initiate required data

        Args:
            buy_fee (Decimal)
            sell_fee (Decimal)
            from_date_str (str)
            to_date_str (str)
            capital (Decimal)
            path (str, optional). Defaults to "data/is/pe_dps.csv".
            index_path (str, optional). Defaults to "data/is/vnindex.csv".
        """
        self.printable = printable
        self.data_service = DataService()
        self.metric = None

        self.inventory = 0
        self.inventory_price = Decimal('0')

        self.daily_assets: List[Decimal] = [capital]
        self.daily_returns: List[Decimal] = []
        self.tracking_dates = []
        self.daily_inventory = []
        self.monthly_inventory = []

        self.cur_date = None
        self.ticker = None
        self.max_order = Decimal("12")
        self.old_timestamp = None
        self.bid_price = None
        self.ask_price = None
        self.ac_loss = Decimal("0.0")
        self.transactions = []
        self.order_logs = []
        self.fibo = [Decimal("1.01"), Decimal("1.01"), Decimal("1.02"), Decimal("1.03"), Decimal("1.05"), Decimal("1.08"), Decimal("1.13"), Decimal("1.21"), Decimal("1.34"), Decimal("1.55"), Decimal("1.89"), Decimal("2.44"), Decimal("3.33"), Decimal("4.77")]

    def move_f1_to_f2(self, f1_price, f2_price):
        """
        TODO: move f1 to f2
        """
        if self.inventory > 0:
            self.ac_loss += (self.inventory_price - f1_price) * 100
            self.inventory_price = f2_price
            self.ac_loss += FEE_PER_CONTRACT * abs(self.inventory)
        elif self.inventory < 0:
            self.ac_loss += (f1_price - self.inventory_price) * 100
            self.inventory_price = f2_price
            self.ac_loss += FEE_PER_CONTRACT * abs(self.inventory)

    def update_pnl(self, close_price: Decimal):
        """
        Daily update pnl

        Args:
            close_price (Decimal)
        """
        cur_asset = self.daily_assets[-1]
        new_asset = None
        if self.inventory == 0:
            new_asset = cur_asset - self.ac_loss
        else:
            sign = 1 if self.inventory > 0 else -1
            pnl = (
                sign * abs(self.inventory) * (close_price - self.inventory_price) * 100
                - self.ac_loss
            )
            new_asset = cur_asset + pnl
            self.inventory_price = close_price

        self.daily_returns.append(new_asset / self.daily_assets[-1] - 1)
        self.daily_assets.append(new_asset)

    def handle_force_sell(self, price: Decimal):
        """
        Handle force sell

        Args:
            price (Decimal): _description_
        """
        while self.get_maximum_placeable(price) < 0:
            sign = 1 if self.inventory < 0 else -1
            self.inventory += sign
            self.ac_loss += abs(price - self.inventory_price) * 100 + FEE_PER_CONTRACT

    def get_maximum_placeable(self, inst_price: Decimal):
        """
        Get maximum placeable

        Args:
            inst_price (Decimal): _description_

        Returns:
            _type_: _description_
        """
        total_placeable = max(
            from_cash_to_tradeable_contracts(
                self.daily_assets[-1] - self.ac_loss, inst_price
            ),
            0,
        )
        return total_placeable - abs(self.inventory)

    def handle_matched_order(self, price):
        """
        Handle matched order

        Args:
            price (_type_): _description_
        """
        matched = 0
        placeable = self.get_maximum_placeable(price)
        if self.bid_price is None or self.ask_price is None:
            return matched

        if self.bid_price >= price and self.inventory >= 0 and placeable > 0 and ((self.inventory + 1) <= 12):
            self.inventory_price = (
                self.inventory_price * abs(self.inventory) + price
            ) / (abs(self.inventory) + 1)
            self.inventory += 1
            matched += 1
        elif self.bid_price >= price and self.inventory < 0 and ((self.inventory + 1) <= 12):
            self.ac_loss -= (self.inventory_price - price) * Decimal(
                '100'
            ) - FEE_PER_CONTRACT
            self.inventory += 1
            matched -= 1

        if self.ask_price <= price and self.inventory <= 0 and placeable > 0 and (abs(self.inventory - 1) <= 12):
            self.inventory_price = (
                self.inventory_price * abs(self.inventory) + price
            ) / (abs(self.inventory) + 1)
            self.inventory -= 1
            matched += 1
        elif self.ask_price <= price and self.inventory > 0 and (abs(self.inventory - 1) <= 12):
            self.ac_loss -= (price - self.inventory_price) * Decimal(
                '100'
            ) - FEE_PER_CONTRACT
            self.inventory -= 1
            matched -= 1
        
        return matched

    def update_bid_ask(self, price: Decimal, step: Decimal, priceEncouragement, timestamp):
        """
        Placing bid ask formula

        Args:
            price (Decimal)
            step (Decimal)
            priceEncouragement (Decimal)
            timestamp (datetime)
        """
        matched = self.handle_matched_order(price)
        
        positionLimit = Decimal("12")
        if self.old_timestamp is None or timestamp > self.old_timestamp + timedelta(
                seconds=int(BACKTESTING_CONFIG["time"])
            ):
            self.old_timestamp = timestamp
            adjustment = (Decimal(self.inventory) / positionLimit) * priceEncouragement 
            fibo_index = 0
            if abs(self.inventory) > 0:
                fibo_index = min(abs(self.inventory) - 1, len(self.fibo) - 1)
                fibo_index = max(fibo_index, 0)  
                adjustment *= self.fibo[fibo_index]
            
            self.bid_price = price - step * (Decimal(1) + adjustment)
            self.ask_price = price + step * (Decimal(1) - adjustment)
            
        elif matched != 0:
            adjustment = (Decimal(self.inventory) / positionLimit) * priceEncouragement
            if abs(self.inventory) > 0:
                fibo_index = min(abs(self.inventory) - 1, len(self.fibo) - 1)
                fibo_index = max(fibo_index, 0)  
                adjustment *= self.fibo[fibo_index]
                
            self.bid_price = price - step * (Decimal(1) + adjustment)
            self.ask_price = price + step * (Decimal(1) - adjustment)

    def process_data(self, evaluation=False):
        """
        Process data with better error handling
        """
        
        prefix_path = "data/os/" if evaluation else "data/is/"
        
        f1_data = pd.read_csv(f"{prefix_path}VN30F1M_data.csv")
        f1_data["datetime"] = pd.to_datetime(
            f1_data["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
        )
        f1_data["date"] = (
            pd.to_datetime(f1_data["date"], format="%Y-%m-%d").copy().dt.date
        )
        f1_data["close"] = f1_data["close"].apply(Decimal)
        f1_data["price"] = f1_data["price"].apply(Decimal)
        f1_data["best-bid"] = f1_data["best-bid"].apply(Decimal)
        f1_data["best-ask"] = f1_data["best-ask"].apply(Decimal)
        f1_data["spread"] = f1_data["spread"].apply(Decimal)

        f2_path = f"{prefix_path}VN30F2M_data.csv"
        # Add validation for numeric columns
        numeric_columns = ["close", "price", "best-bid", "best-ask", "spread"]
        for col in numeric_columns:
            if col in f1_data.columns:
                f1_data[col] = pd.to_numeric(f1_data[col], errors='coerce').apply(Decimal)
                # Check for NaN values
                if f1_data[col].isna().any():
                    print(f"Warning: Found NaN values in {col} column")
                    f1_data = f1_data.dropna(subset=[col])

        f2_data = pd.read_csv(f2_path)
        f2_data = f2_data[["date", "datetime", "tickersymbol", "price", "close"]].copy()
        f2_data["datetime"] = pd.to_datetime(
            f2_data["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
        )
        f2_data["date"] = (
            pd.to_datetime(f2_data["date"], format="%Y-%m-%d").copy().dt.date
        )
        f2_data.rename(columns={"price": "f2_price", "close": "f2_close"}, inplace=True)
        
        # Convert to Decimal with error handling
        f2_data["f2_close"] = pd.to_numeric(f2_data["f2_close"], errors='coerce').apply(Decimal)
        f2_data["f2_price"] = pd.to_numeric(f2_data["f2_price"], errors='coerce').apply(Decimal)
        
        # Drop rows with NaN values
        f2_data = f2_data.dropna(subset=["f2_close", "f2_price"])

        f1_data = pd.merge(
            f1_data,
            f2_data,
            on=["datetime", "date", "tickersymbol"],
            how="outer",
            sort=True,
        )
        f1_data = f1_data.ffill()
        
            
        return f1_data
        

    def run(self, data: pd.DataFrame, step: Decimal, priceEncouragement: Decimal):
        """
        Main backtesting function
        """

        trading_dates = data["date"].unique().tolist()

        start_date = data["datetime"].iloc[0]
        end_date = data["datetime"].iloc[-1]
        expiration_dates = get_expired_dates(start_date, end_date)

        cur_index = 0
        moving_to_f2 = False
        for index, row in data.iterrows():
            self.cur_date = row["datetime"]
            self.ticker = row["tickersymbol"]
            if (
                cur_index != len(trading_dates) - 1
                and not expiration_dates.empty()
                and trading_dates[cur_index + 1] >= expiration_dates.queue[0]
            ):
                self.move_f1_to_f2(row["price"], row["f2_price"])
                expiration_dates.get()
                moving_to_f2 = True
            
            self.handle_force_sell(row["f2_price"] if moving_to_f2 else row["price"])
            self.update_bid_ask(
                row["f2_price"] if moving_to_f2 else row["price"], step, priceEncouragement, row["datetime"]
            )
            if self.inventory > 12:
                continue
            if index == len(data) - 1 or row["date"] != data.iloc[index + 1]["date"]:
                cur_index += 1
                self.update_pnl(row["f2_close"] if moving_to_f2 else row["close"])
                if self.printable:
                    print(
                        f"Realized asset {row['date']}: {int(self.daily_assets[-1] * Decimal('1000'))} VND"
                    )
                moving_to_f2 = False
                self.ac_loss = Decimal("0.0")
                self.bid_price = None
                self.ask_price = None
                self.old_timestamp = None

                self.tracking_dates.append(row["date"])
                self.daily_inventory.append(self.inventory)

        self.metric = Metric(self.daily_returns, None)

    def plot_nav(self, path="result/backtest/nav.png"):
        """
        Plot and save NAV chart to path

        Args:
            path (str, optional): _description_. Defaults to "result/backtest/nav.png".
        """
       
        plt.figure(figsize=(10, 6))

        plt.plot(
            self.tracking_dates,
            self.daily_assets[1:],
            label="Portfolio",
            color='black',
        )

        plt.title('Asset Value Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Asset Value')
        plt.grid(True)
        plt.legend()
        plt.savefig(path, dpi=300, bbox_inches='tight')

    def plot_drawdown(self, path="result/backtest/drawdown.png"):
        """
        Plot and save drawdown chart to path
        Args:
            path (str, optional): _description_. Defaults to "result/backtest/drawdown.png".
        """
        
        
        _, drawdowns = self.metric.maximum_drawdown()

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.tracking_dates,
            drawdowns,
            label="Portfolio",
            color='black',
        )

        plt.title('Draw down Value Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Percentage')
        plt.grid(True)
        plt.savefig(path, dpi=300, bbox_inches='tight')

    def plot_inventory(self, path="result/backtest/inventory.png"):
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.tracking_dates,
            self.daily_inventory,
            label="Portfolio",
            color='black',
        )

        plt.title('Inventory Value Over Time')
        plt.xlabel('Time Step')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    bt = Backtesting(
        capital=Decimal("5e5"),
    )

    data = bt.process_data()
    bt.run(data, Decimal("1.8"), Decimal("0.05"))

    print(
        f"Sharpe ratio: {bt.metric.sharpe_ratio(risk_free_return=Decimal('0.00023')) * Decimal(np.sqrt(250))}"
    )
    print(
        f"Sortino ratio: {bt.metric.sortino_ratio(risk_free_return=Decimal('0.00023')) * Decimal(np.sqrt(250))}"
    )
    mdd, _ = bt.metric.maximum_drawdown()
    print(f"Maximum drawdown: {mdd}")

    bt.plot_nav()
    bt.plot_drawdown()
    bt.plot_inventory()
       