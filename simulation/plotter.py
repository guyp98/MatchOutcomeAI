# plotter.py
import matplotlib.pyplot as plt

class GraphPlotter:
    def __init__(self, team_results, team_name):
        """
        Initialize with the filtered simulation results for a specific team and the team name.
        """
        self.team_results = team_results
        self.team_name = team_name

    def plot_bankroll_and_bets(self):
        """
        Plot two subplots:
          - Bankroll progression over the betting rounds.
          - Bet amount progression over the betting rounds.
        """
        plt.figure(figsize=(12, 8))

        # Subplot 1: Bankroll progression
        plt.subplot(2, 1, 1)
        plt.plot(self.team_results.index, self.team_results['Kelly Bankroll'], label='Kelly Bankroll', marker='o')
        plt.plot(self.team_results.index, self.team_results['Fixed Bankroll'], label='Fixed Bankroll', marker='o')
        plt.plot(self.team_results.index, self.team_results['Martingale Bankroll'], label='Martingale Bankroll', marker='o')
        plt.ylabel("Bankroll")
        plt.title(f"Bankroll Progression for {self.team_name}")
        plt.legend()

        # Subplot 2: Bet amount progression
        plt.subplot(2, 1, 2)
        plt.plot(self.team_results.index, self.team_results['Kelly Bet'], label='Kelly Bet', marker='o')
        plt.plot(self.team_results.index, self.team_results['Fixed Bet'], label='Fixed Bet', marker='o')
        plt.plot(self.team_results.index, self.team_results['Martingale Bet'], label='Martingale Bet', marker='o')
        plt.ylabel("Bet Amount")
        plt.xlabel("Betting Round")
        plt.legend()

        plt.tight_layout()
        plt.show()
