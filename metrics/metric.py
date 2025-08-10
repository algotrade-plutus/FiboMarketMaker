"""
This module is used for calculating metric
"""

from typing import List
from decimal import Decimal
import numpy as np


class Metric:
    """
    Metric: sharpe, sortino, information ratios, MDD
    """

    def __init__(self, period_returns: List[Decimal], benchmark_returns: List[Decimal]):
        """
        Args:
            period_returns (List[Decimal]): _description_
            benchmark_returns (List[Decimal]): _description_
        """
        self.period_returns = period_returns
        self.benchmark_returns = benchmark_returns

    def sharpe_ratio(self, risk_free_return: Decimal) -> Decimal:
        """
        Calculate sharpe ratio

        Args:
            risk_free_return (Decimal): annual risk free

        Raises:
            ValueError: None or empty period returns

        Returns:
            Decimal
        """
        if not self.period_returns:
            raise ValueError('Annual returns should not be None or empty')

        # Calculate excess returns
        excess_returns = [
            period_return - risk_free_return for period_return in self.period_returns
        ]

        return np.mean(excess_returns) / np.std(self.period_returns, ddof=1)

    def sortino_ratio(self, risk_free_return: Decimal) -> Decimal:
        """
        Calculate sortino ratio

        Args:
            risk_free_return (Decimal): _description_

        Raises:
            ValueError: None or empty period returns

        Returns:
            Decimal: _description_
        """
        if not self.period_returns:
            raise ValueError('Annual returns should not be None or empty')

        downside_returns = [
            min(0, period_return - risk_free_return)
            for period_return in self.period_returns
        ]
        downside_risk = np.sqrt(np.mean([d_r**2 for d_r in downside_returns]))

        return (np.mean(self.period_returns) - risk_free_return) / downside_risk

    def maximum_drawdown(self) -> Decimal:
        """
        Calculate maximum drawdown

        Raises:
            ValueError: None or empty period returns
            ValueError: Invalid input

        Returns:
            Decimal: _description_
        """
        dds = []
        if not self.period_returns:
            raise ValueError('Invalid Input')

        if any(period_return <= -1 for period_return in self.period_returns):
            raise ValueError('Invalid Input')

        peak = 1
        cur_perf = 1
        mdd = 0
        for period_return in self.period_returns:
            cur_perf *= 1 + period_return
            peak = max(peak, cur_perf)

            dd = cur_perf / peak - 1
            dds.append(dd)
            mdd = min(dd, mdd)

        return mdd, dds

    def longest_drawdown(self) -> int:
        """
        Calculate longest drawdown

        Raises:
            ValueError: None of empty period returns
            ValueError: Invalid

        Returns:
            int: _description_
        """
        if not self.period_returns:
            raise ValueError('Invalid Input')

        if any(period_return <= -1 for period_return in self.period_returns):
            raise ValueError('Invalid Input')

        cur_period = 0
        max_period = 0
        peak = 1
        cur_perf = 1
        for period_return in self.period_returns:
            cur_perf *= 1 + period_return

            if cur_perf > peak:
                cur_period = 0
                peak = cur_perf
                continue

            cur_period += 1
            max_period = max(max_period, cur_period)

        return max_period

    def information_ratio(self) -> Decimal:
        """
        Calculate informaton ratio

        Raises:
            ValueError: None or empty period return or benchmark return
            ValueError: Not equal length
            ValueError: Invalid input
            ValueError: Invalid length

        Returns:
            Decimal: _description_
        """
        if not self.period_returns or not self.benchmark_returns:
            raise ValueError("Invalid Input")

        if len(self.period_returns) != len(self.benchmark_returns):
            raise ValueError(
                f"Not equal length {len(self.period_returns)} - {len(self.benchmark_returns)}"
            )

        if any(period_return <= -1 for period_return in self.period_returns) or any(
            benchmark_return <= -1 for benchmark_return in self.benchmark_returns
        ):
            raise ValueError("Invalid Input")

        if len(self.period_returns) == 1 or len(self.benchmark_returns) == 1:
            raise ValueError("Invalid length")

        mean_period_returns = np.array(self.period_returns).mean()
        mean_benchmark_returns = np.array(self.benchmark_returns).mean()

        if mean_period_returns == mean_benchmark_returns:
            return 0

        excess_returns = np.array(self.period_returns) - np.array(
            self.benchmark_returns
        )

        return (mean_period_returns - mean_benchmark_returns) / excess_returns.std()
