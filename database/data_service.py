"""
Data service
"""

import psycopg2
import pandas as pd

from database.query import MATCHED_QUERY, BID_ASK_QUERY, CLOSE_QUERY
from config.config import db_params


class DataService:
    """
    Class data service
    """

    def __init__(self) -> None:
        """
        Initiate database secret
        """
        if (
            db_params["host"]
            and db_params["port"]
            and db_params["database"]
            and db_params["user"]
            and db_params["password"]
        ):
            self.connection = psycopg2.connect(**db_params)
            self.is_file = False
        else:
            self.is_file = True

    def get_matched_data(
        self,
        from_year: str,
        to_year: str,
        contract_type: str,
    ) -> pd.DataFrame:
        """
        Get matched data frame

        Args:
            from_year (str)
            to_year (str)

        Returns:
            pd.DataFrame
        """
        cursor = self.connection.cursor()
        cursor.execute(
            MATCHED_QUERY,
            (contract_type, from_year, to_year),
        )

        queries = list(cursor)
        cursor.close()

        columns = ["datetime", "tickersymbol", "price"]
        return pd.DataFrame(queries, columns=columns)

    def get_bid_ask_data(
        self,
        from_date: str,
        to_date: str,
        contract_type: str,
    ) -> pd.DataFrame:
        """
        Get bid ask data frame

        Args:
            from_date (str)
            to_date (str)

        Returns:
            pd.DataFrame
        """
        cursor = self.connection.cursor()
        cursor.execute(BID_ASK_QUERY, (contract_type, from_date, to_date))

        queries = list(cursor)
        cursor.close()

        columns = ["datetime", "tickersymbol", "best-bid", "best-ask", "spread"]
        return pd.DataFrame(queries, columns=columns)

    def get_close_price(
        self,
        from_date: str,
        to_date: str,
        contract_type: str,
    ):
        cursor = self.connection.cursor()
        cursor.execute(CLOSE_QUERY, (contract_type, from_date, to_date))

        queries = list(cursor)
        cursor.close()

        columns = ["date", "tickersymbol", "close"]
        return pd.DataFrame(queries, columns=columns)


data_service = DataService()
