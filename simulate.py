# simulate.py

import pandas as pd
import matplotlib.pyplot as plt
from predictor import simulate_betting
from global_def import *
from data_manager import merge_data



def main():
    #create data
    merge_data()

    # 2. Run the Kelly‐criterion betting simulation
    #    (starts with $100)
    final_bankroll, history = simulate_betting(
        initial_bankroll=100.0
    )

    # 3. Prepare a DataFrame for plotting & analysis
    df = pd.DataFrame({
        'match': range(1, len(history) + 1),
        'bankroll': history
    })

    # 4. Plot bankroll over time
    plt.figure()
    plt.plot(df['match'], df['bankroll'])
    plt.title(f"Bankroll Over Season {simulate_year}")
    plt.xlabel("Match Number")
    plt.ylabel("Bankroll ($)")
    plt.tight_layout()
    plt.savefig(f"bankroll_season_{simulate_year}.png")
    plt.close()

    # 5. Compute risk statistics
    returns = df['bankroll'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = mean_return / std_return if std_return != 0 else float('nan')

    cum_max = df['bankroll'].cummax()
    drawdown = (df['bankroll'] - cum_max) / cum_max
    max_drawdown = drawdown.min()

    # 6. Print out the results
    print(f"\n=== Simulation Results for Season {simulate_year} ===")
    print(f"Final bankroll:        ${final_bankroll:.2f}")
    print(f"Mean return/match:     {mean_return:.2%}")
    print(f"Return volatility:     {std_return:.2%}")
    print(f"Sharpe ratio:          {sharpe_ratio:.2f}")
    print(f"Max drawdown:          {max_drawdown:.2%}\n")

if __name__ == "__main__":
    main()
