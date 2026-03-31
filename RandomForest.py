import numpy as np
import pandas as pd
import matplotlib as mtl

def run_random_forest(df):
    #TODO: Implement random forest algorithm
    pass

def format_properly(result):
    #TODO: Print output properly based on result
    pass

def main():
    df = pd.read_csv("./database/All_Pokemon.csv")

    result = run_random_forest(df)

    print(format_properly(result))

if __name__ == '__main__':
    main()
