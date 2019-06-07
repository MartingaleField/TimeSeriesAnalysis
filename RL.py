import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime

from pgportfolio.tools.configprocess import preprocess_config
from pgportfolio.tools.configprocess import load_config
from pgportfolio.tools.trade import save_test_data
from pgportfolio.tools.shortcut import execute_backtest
from pgportfolio.resultprocess import plot

from pgportfolio.marketdata.datamatrices import DataMatrices

with open("./pgportfolio/net_config.json") as file:
    config = json.load(file)
config = preprocess_config(config)
start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
DataMatrices(start=start,
             end=end,
             feature_number=config["input"]["feature_number"],
             window_size=config["input"]["window_size"],
             online=True,
             period=config["input"]["global_period"],
             volume_average_days=config["input"]["volume_average_days"],
             coin_filter=config["input"]["coin_number"],
             is_permed=config["input"]["is_permed"],
             test_portion=config["input"]["test_portion"],
             portion_reversed=config["input"]["portion_reversed"])
