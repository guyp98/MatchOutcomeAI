import os
from string import capwords
from predictor import get_teams_from_season, predictor, output_previous_prediction
from cli import clear, selection, log, log_invalid_selection
from global_def import *
from data_manager import *

def pause():
    os.system('pause')
    clear()

def interface():
    while True:
        module = selection()
        if module == "0":
            quit()
        elif module == "1":
            clear()
            scrape_data(
                [
                    "2016",
                    "2017",
                    "2018",
                    "2019",
                    "2020",
                    "2021",
                    "2022"
                    ]
                )
            pause()
        elif module == "2":
            clear()
            scrape_data(
                [
                    "2022"
                    ]
                )
            pause()
        elif module == "3":
            home_team, away_team = prediction_interface()
            predictor(home_team, away_team)
            pause()
        elif module == "4":
            output_previous_prediction()
            pause()
        else:
            log_invalid_selection()

def prediction_interface():
    teams = get_teams_from_season(season=2022)
    while True:
        clear()

        print("AVAILABLE TEAMS\n")

        log(teams)

        home_team = capwords(input("ENTER HOME TEAM\n"))
        if home_team not in teams:
            log_invalid_selection()
            continue

        away_team = capwords(input("\nENTER AWAY TEAM\n"))
        if away_team not in teams:
            log_invalid_selection()
            continue

        if home_team == away_team:
            log_invalid_selection()
            continue

        print("")

        return home_team, away_team

if __name__ == "__main__":
    interface()