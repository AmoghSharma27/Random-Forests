from RandomForest import RandomForests
import pandas as pd
import math
import time


def import_csv(file_name="./database/All_Pokemon.csv"):
    return pd.read_csv(file_name)


def main():
    data = import_csv()
    start_time = time.time()

    # 70-30 split in testing and training data
    num_rows = len(data)
    train_data = data.head(math.ceil(num_rows * 0.7))
    test_data = data.tail(math.floor(num_rows * 0.3))

    random_forests = RandomForests(train_data, 50)

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
            print(row_end - row_start)
        else:
            print("Pokemon:\t", test_data.iloc[i]["Name"])
            print("Actual:\t", test_data.iloc[i]["Type 1"])
            print("Predicted:\t", pred)
        amount += 1

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Runtime:\t{total_time:.4f} seconds")

    # Optional: average prediction time per row
    avg_row_time = sum(row_times) / len(row_times)
    print(f"Avg Prediction Time per Row:\t{avg_row_time:.6f} seconds")
    print("Amount:\t", amount)
    print(f"Accuracy:\t{(correct / amount) * 100.00}%")


if __name__ == "__main__":
    main()
