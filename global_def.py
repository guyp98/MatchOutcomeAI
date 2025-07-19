import os
path_to_csv_data = "./data/csv_datasets/epl/"
path_to_raw_data = "./data/raw_datasets/epl/"
os.makedirs(path_to_csv_data, exist_ok=True)
os.makedirs(path_to_raw_data, exist_ok=True)
all_data_path = path_to_csv_data + "all_seasons.csv"
years_for_training = ["2016", "2017", "2018", "2019", "2020", "2021"]
simulate_year = "2022"