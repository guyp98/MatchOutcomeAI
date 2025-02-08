# main.py
import pandas as pd
from betting_simulation import BettingSimulation
from plotter import GraphPlotter

def main():
    # Path to your CSV file
    csv_file = "../data/csv_datasets/epl/2016_season.csv"
    
    # Create and run the betting simulation
    simulation = BettingSimulation(csv_file)
    simulation.simulate_betting()

    # Retrieve simulation results and display a preview.
    results = simulation.get_results()
    print("Simulation results (first 5 rows):")
    print(results.head())

    # Identify the team with the highest overall draw ratio.
    highest_draw_team, highest_ratio = simulation.get_team_with_highest_draw_ratio()
    print(f"\nTeam with highest draw ratio: {highest_draw_team} (ratio: {highest_ratio:.2f})")

    # Filter results for the selected team (assuming 'Team' column corresponds to the home team).
    team_results = results[results['Team'] == highest_draw_team].reset_index(drop=True)

    # Plot the simulation results for the selected team.
    plotter = GraphPlotter(team_results, highest_draw_team)
    plotter.plot_bankroll_and_bets()

if __name__ == "__main__":
    main()
