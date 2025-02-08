# betting_simulation.py
import pandas as pd

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
        # Martingale bet may vary per round:
        self.martingale_bet = fixed_bet  
        # This list will store the simulation results (bankroll and bet amounts)
        self.bet_results = []
        self.process_data()

    def process_data(self):
        """
        Process the input data to compute each match's full-time result (FTR).
        'D' = Draw, 'H' = Home win, 'A' = Away win.
        """
        self.data['FTR'] = self.data.apply(
            lambda row: 'D' if row['home_goals'] == row['away_goals'] 
                        else ('H' if row['home_goals'] > row['away_goals'] else 'A'),
            axis=1
        )

    def calculate_draw_probability(self, team, num_matches=20):
        """
        Calculate a team's draw probability based on its most recent `num_matches`.
        """
        team_games = self.data[
            (self.data['home_team'] == team) | (self.data['away_team'] == team)
        ].tail(num_matches)
        num_draws = (team_games['FTR'] == 'D').sum()
        return num_draws / len(team_games) if len(team_games) > 0 else 0

    def kelly_bet(self, bankroll, p):
        """
        Compute the Kelly Criterion wager based on the current bankroll and probability p.
        """
        b = self.odds - 1
        kelly_fraction = (b * p - (1 - p)) / b
        return max(0, kelly_fraction * bankroll)

    def simulate_betting(self):
        """
        Run the simulation over every match in the dataset.
        Records bankroll progression and wager amounts for:
          - Kelly Criterion Betting
          - Fixed Stake Betting
          - Martingale Betting
        """
        for _, game in self.data.iterrows():
            team = game['home_team']  # We assume betting on the home team's draw.
            p_draw = self.calculate_draw_probability(team)

            # ----- Kelly Criterion Betting -----
            kelly_wager = self.kelly_bet(self.kelly_bankroll, p_draw)
            current_kelly_bet = kelly_wager  # wager for this round
            if game['FTR'] == 'D':
                self.kelly_bankroll += kelly_wager * (self.odds - 1)
            else:
                self.kelly_bankroll -= kelly_wager

            # ----- Fixed Stake Betting -----
            current_fixed_bet = self.fixed_bet
            if game['FTR'] == 'D':
                self.fixed_bankroll += self.fixed_bet * (self.odds - 1)
            else:
                self.fixed_bankroll -= self.fixed_bet

            # ----- Martingale Betting -----
            current_martingale_bet = self.martingale_bet
            if game['FTR'] == 'D':
                self.martingale_bankroll += self.martingale_bet * (self.odds - 1)
                self.martingale_bet = self.fixed_bet  # reset after win
            else:
                self.martingale_bankroll -= self.martingale_bet
                # Double the bet up to a maximum limit.
                self.martingale_bet = min(
                    self.martingale_bet * 2,
                    self.fixed_bet * (2 ** self.max_martingale_streak)
                )

            # Record results for this round:
            self.bet_results.append([
                team, p_draw,
                self.kelly_bankroll, current_kelly_bet,
                self.fixed_bankroll, current_fixed_bet,
                self.martingale_bankroll, current_martingale_bet
            ])

    def get_results(self):
        """
        Return the simulation results as a pandas DataFrame.
        The DataFrame includes:
          - 'Team', 'P_Draw', 'Kelly Bankroll', 'Kelly Bet',
            'Fixed Bankroll', 'Fixed Bet', 'Martingale Bankroll', 'Martingale Bet'
        """
        return pd.DataFrame(self.bet_results, columns=[
            'Team', 'P_Draw',
            'Kelly Bankroll', 'Kelly Bet',
            'Fixed Bankroll', 'Fixed Bet',
            'Martingale Bankroll', 'Martingale Bet'
        ])

    def get_team_with_highest_draw_ratio(self):
        """
        Determine which team has the highest overall draw ratio.
        The ratio is computed as: (total draws) / (total matches played) (both home and away).
        Returns:
            tuple: (team_name, draw_ratio)
        """
        teams = pd.unique(self.data[['home_team', 'away_team']].values.ravel('K'))
        team_draw_ratios = {}
        for team in teams:
            team_games = self.data[
                (self.data['home_team'] == team) | (self.data['away_team'] == team)
            ]
            if len(team_games) > 0:
                draw_ratio = (team_games['FTR'] == 'D').sum() / len(team_games)
            else:
                draw_ratio = 0
            team_draw_ratios[team] = draw_ratio
        highest_draw_team = max(team_draw_ratios, key=team_draw_ratios.get)
        return highest_draw_team, team_draw_ratios[highest_draw_team]
