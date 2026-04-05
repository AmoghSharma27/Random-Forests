from RandomForest import RandomForests
import pandas as pd
import numpy as np
import time

ratio = 0.7
num_trees = 10
random_forests = None


def import_csv(file_name="./database/All_Pokemon.csv"):
    return pd.read_csv(file_name)


def main():
    global random_forests, ratio
    random_forests = None  # Reset to force retraining
    data = import_csv()
    start_time = time.time()

    # 70-30 split in testing and training data
    """train_data = data.head(math.ceil(num_rows * ratio))
    test_data = data.tail(math.floor(num_rows * (1 - ratio)))"""
    train_data = data.sample(frac=ratio)
    test_data = data.drop(train_data.index)

    train_start_time = time.time()
    random_forests = RandomForests(train_data, num_trees)
    train_end_time = time.time()

    # Calculate how correct
    correct = 0
    amount = 0
    row_times = []
    for i in range(len(test_data)):

        row_start = time.time()
        pred = random_forests.predict(test_data.iloc[i])
        row_end = time.time()

        row_times.append(row_end - row_start)
        if pred == test_data.iloc[i]["Type 1"]:
            correct += 1

        amount += 1

    end_time = time.time()
    total_time = end_time - start_time

    avg_row_time = sum(row_times) / len(row_times)

    print("Training Time:\t", train_end_time - train_start_time)
    print(f"Avg Prediction Time per Row:\t{avg_row_time:.6f} seconds")
    print(f"Total Runtime:\t{total_time:.4f} seconds")
    print(f"Accuracy:\t{(correct / amount) * 100.00}%")

    return (correct / amount) * 100.00


def run_main(acc):
    global num_trees, ratio
    best_accuracy = acc
    best_num_trees = 10
    best_ratio = 0.7

    for nt in range(10, 101, 10):
        for r in np.arange(0.6, 0.91, 0.1):
            num_trees = nt
            ratio = r
            accuracy = main()
            if accuracy > best_accuracy:
                print(
                    f"THIS WAS BETTER!\tprev_num_trees:{best_num_trees}\tprev_ratio:{best_ratio}\nprev_accuracy:{best_accuracy}\t"
                )
                best_accuracy = accuracy
                best_num_trees = nt
                best_ratio = r

            print(f"curr num_trees:\t{num_trees}\tcurr ratio:\t{ratio}\nAccuracy:\t{accuracy:.2f}%")

    print("_" * 50)
    print(f"Best accuracy:\t{best_accuracy:.2f}%")
    print(f"Best ratio:\t{best_ratio:.2f}")
    print(f"Best num_trees:\t{best_num_trees}")
    print("_" * 50)


if __name__ == "__main__":
    acc = main()
    run_main(acc)
