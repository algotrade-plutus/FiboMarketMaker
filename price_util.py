import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("data/is/VN30F1M_data.csv")

    dates = ["2022-03-29"]
    path = "result/backtest/f1_price.png"
    df = df[df["date"].isin(dates)]
    df["datetime"] = pd.to_datetime(df["datetime"])
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["datetime"],
        df["price"],
        label="matched price",
        color='black',
    )

    plt.title('VN30F1M price')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
