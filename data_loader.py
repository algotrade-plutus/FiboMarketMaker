"""
Loading data to csv file
"""

import os
from datetime import datetime
import pandas as pd
from database.data_service import DataService
from config.config import BACKTESTING_CONFIG


def init_folder(path: str):
    """
    Creates the folder if it does not exist.

    Args:
        path (str): Path to the folder you want to initialize.
    """
    os.makedirs(path, exist_ok=True)


def loading_bid_ask(from_date, to_date, contract_type, validation=False):
    data_service = DataService()

    print("Loading close price data...")
    close_price = data_service.get_close_price(from_date, to_date, contract_type)
    close_price["date"] = (
        pd.to_datetime(close_price["date"], format="%Y-%m-%d").copy().dt.date
    )

    print(f"Loading {contract_type} bid-ask data...")
    bid_ask = data_service.get_bid_ask_data(from_date, to_date, contract_type)
    bid_ask = bid_ask.astype({"best-bid": float, "best-ask": float, "spread": float})
    bid_ask["datetime"] = pd.to_datetime(
        bid_ask["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
    )

    print("Loading matched data...")
    matched = data_service.get_matched_data(from_date, to_date, contract_type)
    matched = matched.astype({"price": float})
    matched["datetime"] = pd.to_datetime(
        matched["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
    )

    data = pd.merge(
        matched, bid_ask, on=["datetime", "tickersymbol"], how="outer", sort=True
    )
    data = data.ffill()
    data["date"] = data["datetime"].copy().dt.date

    is_data = pd.merge(
        data, close_price, on=["date", "tickersymbol"], how="inner", sort=True
    )
    is_data.to_csv(
        (
            f"data/os/{contract_type}_data.csv"
            if validation
            else f"data/is/{contract_type}_data.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    required_directories = [
        "data",
        "data/is",
        "data/os",
        "result/optimization",
        "result/backtest",
        "result/optimization",
    ]
    for dr in required_directories:
        init_folder(dr)

    is_from_date_str = BACKTESTING_CONFIG["is_from_date_str"]
    is_to_date_str = BACKTESTING_CONFIG["is_end_date_str"]
    os_from_date_str = BACKTESTING_CONFIG["os_from_date_str"]
    os_to_date_str = BACKTESTING_CONFIG["os_end_date_str"]
    is_from_date = datetime.strptime(is_from_date_str, "%Y-%m-%d %H:%M:%S").date()
    is_to_date = datetime.strptime(is_to_date_str, "%Y-%m-%d %H:%M:%S").date()
    os_from_date = datetime.strptime(os_from_date_str, "%Y-%m-%d %H:%M:%S").date()
    os_to_date = datetime.strptime(os_to_date_str, "%Y-%m-%d %H:%M:%S").date()

    print("Loading in-sample data")
    loading_bid_ask(is_from_date, is_to_date, "VN30F1M")
    loading_bid_ask(is_from_date, is_to_date, "VN30F2M")

    print("Loading out-sample data")
    loading_bid_ask(os_from_date, os_to_date, "VN30F1M", validation=True)
    loading_bid_ask(os_from_date, os_to_date, "VN30F2M", validation=True)
