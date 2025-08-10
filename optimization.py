"""
Optimization module
"""

import numpy as np
from decimal import Decimal
import logging
import optuna
from optuna.samplers import TPESampler
from config.config import OPTIMIZATION_CONFIG
from backtesting import Backtesting


class OptunaCallBack:
    """
    Optuna call back class
    """

    def __init__(self, backtesting_instance) -> None:
        """
        Init optuna callback
        
        Args:
            backtesting_instance: The backtesting instance to access metrics
        """
        logging.basicConfig(
            filename="result/optimization/optimization.log.csv",
            format="%(message)s",
            filemode="w",
        )
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        self.logger = logger
        self.bt = backtesting_instance
        self.logger.info("number,step,priceEncouragement,sharpe_ratio,mdd,valid")

    def __call__(self, _: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Log trial results

        Args:
            study (optuna.study.Study): _description_
            trial (optuna.trial.FrozenTrial): _description_
        """
        step = trial.params["step"]
        priceEncouragement = trial.params["priceEncouragement"]
        
        # Extract additional metrics from the last run
        try:
            sharpe_ratio = self.bt.metric.sharpe_ratio(risk_free_return=Decimal('0.00023')) * Decimal(np.sqrt(250))
            mdd, _ = self.bt.metric.maximum_drawdown()
            # Check if MDD constraint is satisfied
            valid = abs(float(mdd)) <= 0.20  # MDD dưới 20%
            self.logger.info(
                "%s,%s,%s,%s,%s,%s",
                trial.number,
                step,
                priceEncouragement,
                float(sharpe_ratio),
                float(mdd),
                valid,
            )
        except Exception as e:
            # Fallback to original logging if metrics are not available
            self.logger.info(
                "%s,%s,%s,,,False",
                trial.number,
                step,
                priceEncouragement,
            )


if __name__ == "__main__":
    # Load data once
    bt_temp = Backtesting(capital=Decimal("5e5"), printable=False)
    data = bt_temp.process_data()

    def objective(trial):
        """
        Objective function: Maximize Sharpe ratio with MDD constraint
        """
        # Use suggest_float instead of suggest_int for step parameter
        step = trial.suggest_float("step", OPTIMIZATION_CONFIG["step"][0], OPTIMIZATION_CONFIG["step"][1], step=0.1)
        priceEncouragement = trial.suggest_float(
            "priceEncouragement", OPTIMIZATION_CONFIG["priceEncouragement"][0], OPTIMIZATION_CONFIG["priceEncouragement"][1], step=0.01
        )
        
        # Create new backtesting instance for each trial to ensure clean state
        bt = Backtesting(capital=Decimal("5e5"), printable=False)
        bt.run(data, Decimal(step), Decimal(priceEncouragement))
        
        sharpe_ratio = bt.metric.sharpe_ratio(risk_free_return=Decimal('0.00023')) * Decimal(np.sqrt(250))
        mdd, _ = bt.metric.maximum_drawdown()
        
        # Store the bt instance for callback access
        objective.bt = bt
        
        # Ràng buộc: MDD phải dưới 20%
        if abs(float(mdd)) > 0.20:
            # Nếu MDD > 20%, trả về giá trị rất thấp để loại bỏ
            raise optuna.TrialPruned()
        
        # Trả về Sharpe ratio để maximize
        return float(sharpe_ratio)

    # Create callback with a dummy bt instance (will be updated in objective)
    dummy_bt = Backtesting(capital=Decimal("5e5"), printable=False)
    optunaCallBack = OptunaCallBack(dummy_bt)
    
    # Update callback to use the current bt instance
    def updated_callback(study, trial):
        if hasattr(objective, 'bt'):
            optunaCallBack.bt = objective.bt
        optunaCallBack(study, trial)

    # Create single-objective study (maximize Sharpe ratio)
    study = optuna.create_study(
        sampler=TPESampler(seed=OPTIMIZATION_CONFIG["random_seed"]),
        direction="maximize",  # maximize sharpe_ratio
        pruner=optuna.pruners.MedianPruner()  # Thêm pruner để loại bỏ trial không tốt
    )
    
    study.optimize(
        objective, 
        n_trials=OPTIMIZATION_CONFIG["no_trials"], 
        callbacks=[updated_callback],
        show_progress_bar=True
    )
    
    # Print best results
    print("\nOptimization completed!")
    print(f"Best trial:")
    trial = study.best_trial
    print(f"  Sharpe Ratio: {trial.value:.4f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Run best trial again to get MDD
    bt_best = Backtesting(capital=Decimal("5e5"), printable=False)
    bt_best.run(data, Decimal(trial.params["step"]), Decimal(trial.params["priceEncouragement"]))
    mdd_best, _ = bt_best.metric.maximum_drawdown()
    print(f"  MDD: {float(mdd_best):.4f} ({float(mdd_best)*100:.2f}%)")
    
    # Print statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\nStatistics:")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials (MDD > 20%): {len(pruned_trials)}")
    print(f"  Total trials: {len(study.trials)}")