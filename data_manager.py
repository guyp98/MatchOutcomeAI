import os
import pandas as pd
from global_def import *
from cli import log
from data_scraper.api_data_scraper import Scraper
import data_scraper.json_data_processor as json_data_processor


def scrape_data(seasons: list):
    
    log("Scraping data from the API...")

    for season in seasons:
        scraper = Scraper(season)
        scraper.scrape()

    log("Data has been scraped for the specified seasons")

    output_data()
    merge_data()

    return

def output_data():

    log("Processing Data...")

    files = os.listdir(path_to_csv_data)

    for file in files:

        file_name = os.path.join(file).split(".")[0]

        data = json_data_processor.load_raw_dataset(file_name)
        json_data_processor.output_dataset(file_name, data)

    return

def merge_data():

    log("Data has been processed and output")

    files = os.listdir(path_to_csv_data)

    df_concat = pd.concat([pd.read_csv(path_to_csv_data + f"{file}") for file in files if file != "all_seasons.csv"], ignore_index=True)
    df_concat.to_csv(all_data_path, index=False)

    df_concat_train = pd.concat([ pd.read_csv(path_to_csv_data + f"{file}") for file in files for year in years_for_training if year in file], ignore_index=True)
    df_concat_train.to_csv(path_to_csv_data + "train_seasons.csv", index=False)
    
    return