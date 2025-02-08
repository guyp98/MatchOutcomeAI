import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

class BettingSimulation:
    def __init__(self, csv_file, initial_bankroll=1000, fixed_bet=10, max_martingale_streak=5, odds=3.2):
        self.data = pd.read_csv(csv_file)
        self.initial_bankroll = initial_bankroll
        self.fixed_bet = fixed_bet
        self.max_martingale_streak = max_martingale_streak
        self.odds = odds
        # Initialize bankrolls for the three strategies
        self.kelly_bankroll = initial_bankroll
        self.fixed_bankroll = initial_bankroll
        self.martingale_bankroll = initial_bankroll
        # For Martingale, the bet may change from round to round.
        self.martingale_bet = fixed_bet  
        # This list will store the results (and bet amounts) for each match.
        self.bet_results = []
        self.process_data()

    def process_data(self):
        """
        Compute the full-time result for each match based on goals.
        'D' = Draw, 'H' = Home win, 'A' = Away win.
        """
        self.data['FTR'] = self.data.apply(
            lambda row: 'D' if row['home_goals'] == row['away_goals'] 
                        else ('H' if row['home_goals'] > row['away_goals'] else 'A'),
            axis=1
        )
    
    def calculate_draw_probability(self, team, num_matches=20):
        """
        Calculate a team’s draw probability based on its most recent `num_matches` matches.
        """
        team_games = self.data[(self.data['home_team'] == team) | (self.data['away_team'] == team)].tail(num_matches)
        num_draws = (team_games['FTR'] == 'D').sum()
        return num_draws / len(team_games) if len(team_games) > 0 else 0

    def kelly_bet(self, bankroll, p):
        """
        Compute the Kelly Criterion bet amount given the current bankroll and probability p.
        """
        b = self.odds - 1
        kelly_fraction = (b * p - (1 - p)) / b
        return max(0, kelly_fraction * bankroll)

    def simulate_betting(self):
        """
        Run the betting simulation over every match in the dataset.
        For each match, we record the current bankroll and the bet amount for each strategy.
        Note: In this simulation, we “bet” on the home team’s draw outcome.
        """
        for _, game in self.data.iterrows():
            team = game['home_team']  # we assume betting on the home team
            p_draw = self.calculate_draw_probability(team)
            
            # ----- Kelly Criterion Betting -----
            kelly_wager = self.kelly_bet(self.kelly_bankroll, p_draw)
            current_kelly_bet = kelly_wager  # record the wager for this round
            if game['FTR'] == 'D':
                self.kelly_bankroll += kelly_wager * (self.odds - 1)
            else:
                self.kelly_bankroll -= kelly_wager
            
            # ----- Fixed Stake Betting -----
            current_fixed_bet = self.fixed_bet  # always the same fixed bet
            if game['FTR'] == 'D':
                self.fixed_bankroll += self.fixed_bet * (self.odds - 1)
            else:
                self.fixed_bankroll -= self.fixed_bet
            
            # ----- Martingale Betting -----
            # Record the bet amount before potentially updating it.
            current_martingale_bet = self.martingale_bet  
            if game['FTR'] == 'D':
                self.martingale_bankroll += self.martingale_bet * (self.odds - 1)
                self.martingale_bet = self.fixed_bet  # reset after a win
            else:
                self.martingale_bankroll -= self.martingale_bet
                # Double the bet, but do not exceed a maximum limit
                self.martingale_bet = min(
                    self.martingale_bet * 2, 
                    self.fixed_bet * (2**self.max_martingale_streak)
                )
            
            # Append all information: team name, draw probability, bankrolls, and bet amounts.
            self.bet_results.append([
                team, p_draw, 
                self.kelly_bankroll, current_kelly_bet,
                self.fixed_bankroll, current_fixed_bet,
                self.martingale_bankroll, current_martingale_bet
            ])

    def get_results(self):
        """
        Return the simulation results as a DataFrame.
        The columns include:
          - 'Team': the home team for that match,
          - 'P_Draw': the calculated draw probability,
          - 'Kelly Bankroll' and 'Kelly Bet': bankroll and wager for Kelly betting,
          - 'Fixed Bankroll' and 'Fixed Bet': bankroll and wager for fixed stake betting,
          - 'Martingale Bankroll' and 'Martingale Bet': bankroll and wager for Martingale betting.
        """
        return pd.DataFrame(self.bet_results, columns=[
            'Team', 'P_Draw', 
            'Kelly Bankroll', 'Kelly Bet',
            'Fixed Bankroll', 'Fixed Bet',
            'Martingale Bankroll', 'Martingale Bet'
        ])

    def get_team_with_highest_draw_ratio(self):
        """
        Determine which team in the dataset has the highest overall draw ratio.
        The draw ratio is computed as: (total draws) / (total matches played) 
        (considering both home and away games).
        Returns:
            (team_name, draw_ratio)
        """
        # Get the unique teams from both home and away columns.
        teams = pd.unique(self.data[['home_team', 'away_team']].values.ravel('K'))
        team_draw_ratios = {}
        for team in teams:
            team_games = self.data[(self.data['home_team'] == team) | (self.data['away_team'] == team)]
            if len(team_games) > 0:
                draw_ratio = (team_games['FTR'] == 'D').sum() / len(team_games)
            else:
                draw_ratio = 0
            team_draw_ratios[team] = draw_ratio
        highest_draw_team = max(team_draw_ratios, key=team_draw_ratios.get)
        return highest_draw_team, team_draw_ratios[highest_draw_team]

# ------------------------------
# Run the simulation
# ------------------------------

csv_file = "./data/csv_datasets/epl/2016_season.csv"
simulation = BettingSimulation(csv_file)
simulation.simulate_betting()

# Get all simulation results as a DataFrame
results = simulation.get_results()
print(results)

# Identify the team with the highest overall draw ratio.
highest_draw_team, highest_ratio = simulation.get_team_with_highest_draw_ratio()
print(f"Team with highest draw ratio: {highest_draw_team} (ratio: {highest_ratio:.2f})")

# ------------------------------
# Filter the results for the selected team.
# ------------------------------
# (Note: In the simulation we use the home_team; therefore, here we filter for matches where the
#  home team equals the team with the highest overall draw ratio.)
team_results = results[results['Team'] == highest_draw_team].reset_index(drop=True)

# ------------------------------
# Plot the bankroll and bet amounts over the simulation rounds.
# ------------------------------

plt.figure(figsize=(12, 8))

# First subplot: bankroll progression
plt.subplot(2, 1, 1)
plt.plot(team_results.index, team_results['Kelly Bankroll'], label='Kelly Bankroll', marker='o')
plt.plot(team_results.index, team_results['Fixed Bankroll'], label='Fixed Bankroll', marker='o')
plt.plot(team_results.index, team_results['Martingale Bankroll'], label='Martingale Bankroll', marker='o')
plt.ylabel("Bankroll")
plt.title(f"Bankroll Progression for {highest_draw_team}")
plt.legend()

# Second subplot: bet amount progression
plt.subplot(2, 1, 2)
plt.plot(team_results.index, team_results['Kelly Bet'], label='Kelly Bet', marker='o')
plt.plot(team_results.index, team_results['Fixed Bet'], label='Fixed Bet', marker='o')
plt.plot(team_results.index, team_results['Martingale Bet'], label='Martingale Bet', marker='o')
plt.ylabel("Bet Amount")
plt.xlabel("Betting Round")
plt.legend()

plt.tight_layout()
plt.show()
