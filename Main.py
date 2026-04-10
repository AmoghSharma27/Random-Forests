from RandomForest import RandomForests
import pandas as pd
import time

ratio = 0.75
num_trees = 11
target = "Type 1"
all_times = {}

def import_csv(file_name="./database/All_Pokemon.csv"):
    return pd.read_csv(file_name)


def main():
    global num_trees, ratio, target
    data = import_csv()
    start_time = time.time()

    train_start_time = time.time()
    # split testing and training data based on the ratio parameter
    train_data = data.sample(frac=ratio)
    test_data = data.drop(train_data.index)
    random_forests = RandomForests(train_data=train_data, num_trees=num_trees, target=target)
    train_end_time = time.time()

    # Calculate how correct our predictions are
    correct = 0
    amount = 0
    for i in range(len(test_data)):
        pred = random_forests.predict(test_data.iloc[i])
        if target != "Type 1":
            if pred == test_data.iloc[i][target]:
                correct += 1
        elif pred == test_data.iloc[i][target] or pred == test_data.iloc[i]["Type 2"]:
            correct += 1
        amount += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("Training Time:\t", train_end_time - train_start_time)
    print(f"Total Runtime:\t{total_time:.4f} seconds")
    print(f"Accuracy:\t{(correct / amount) * 100.00:.2f}%")

    return (correct / amount) * 100.00, total_time

if __name__ == "__main__":
    main()
