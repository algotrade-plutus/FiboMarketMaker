"""
This module provides helper functions
"""

from datetime import datetime
from decimal import Decimal
from queue import Queue
from dateutil.rrule import rrule, MONTHLY, TH


def from_cash_to_tradeable_contracts(
    cash: Decimal,
    inst_price: Decimal,
    multiplier=Decimal("100"),
    margin_rate=Decimal("0.17"),
) -> int:
    """
    Get total tradable contracts

    Args:
        cash (Decimal): _description_
        inst_price (Decimal): _description_
        multiplier (int, optional): _description_. Defaults to 100.
        margin_rate (float, optional): _description_. Defaults to 0.17.

    Returns:
        int: _description_
    """
    return int(cash / (inst_price * multiplier * margin_rate))


def get_expired_dates(start_date: datetime, end_date: datetime) -> Queue:
    """
    Get estimated expiration dates

    Args:
        start_date (datetime)
        end_date (datetime)

    Returns:
        Queue
    """
    third_thursdays = list(
        rrule(
            freq=MONTHLY,
            byweekday=TH(3),
            dtstart=start_date,
            until=end_date,
        )
    )

    queue = Queue()
    for d in third_thursdays:
        queue.put(d.date())

    return queue
