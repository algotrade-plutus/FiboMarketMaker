"""
Out-sample evaluation module
"""

from decimal import Decimal
from config.config import BEST_CONFIG
from backtesting import Backtesting
import numpy as np


if __name__ == "__main__":
    bt = Backtesting(capital=Decimal('5e5'))

    data = bt.process_data(evaluation=True)
    bt.run(data, Decimal(BEST_CONFIG["step"]), Decimal(BEST_CONFIG["priceEncouragement"]))
    bt.plot_nav(path="result/optimization/nav.png")
    bt.plot_drawdown(path="result/optimization/drawdown.png")
    bt.plot_inventory(path="result/optimization/inventory.png")
    print(
        f"Sharpe ratio: {bt.metric.sharpe_ratio(risk_free_return=Decimal('0.00023')) * Decimal(np.sqrt(250))}"
    )
    print(
        f"Sortino ratio: {bt.metric.sortino_ratio(risk_free_return=Decimal('0.00023')) * Decimal(np.sqrt(250))}"
    )
    mdd, _ = bt.metric.maximum_drawdown()
    print(f"Maximum drawdown: {mdd}")
