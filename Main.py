from RandomForest import RandomForests
import pandas as pd
import matplotlib.pyplot as plt
import time

ratio = 0.77
num_trees = 11
target = "Type 1"
(random_f, train_d, num_t, t) = (None, None, None, None)
flag = False

def import_csv(file_name="./database/All_Pokemon.csv"):
    return pd.read_csv(file_name)


def main():
    global num_trees, ratio, target, random_f, flag, train_d, num_t, t
    data = import_csv()
    start_time = time.time()

    # split testing and training data based on the ratio parameter
    train_data = data.sample(frac=ratio)
    test_data = data.drop(train_data.index)

    train_start_time = time.time()

    if flag:
        random_forests = RandomForests(train_data=train_data, num_trees=num_trees, target=target)
        (random_f, train_d, num_t, t) = random_forests.get_forest()
        flag = False
    else:
        random_forests =  RandomForests(train_data=train_d, num_trees=num_t, target=t, forest=random_f)

    train_end_time = time.time()

    # Calculate how correct our predictions are
    correct = 0
    amount = 0
    for i in range(len(test_data)):
        pred = random_forests.predict(test_data.iloc[i])
        if pred == test_data.iloc[i][target]:
            correct += 1
        amount += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("Training Time:\t", train_end_time - train_start_time)
    print(f"Total Runtime:\t{total_time:.4f} seconds")
    print(f"Accuracy:\t{(correct / amount) * 100.00:.2f}%")

    return (correct / amount) * 100.00

def run_main():
    global num_trees, ratio, target, flag

    accuracies = []
    num_trees_list = []

    for i in range(1, 21):
        flag = True
        num_trees_list.append(i)
        num_trees = i
        acc = 0

        for j in range(11):
            acc += main()
            flag = False
        acc /= 11

        accuracies.append(acc)

    plt.plot(num_trees_list, accuracies)
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy (in %)")
    plt.title("Accuracy vs Number of Trees")
    plt.show()

    best_acc = 0
    best_i = 0
    for i, acc in enumerate(accuracies):
        if acc > best_acc:
            best_acc = acc
            best_i = i

    num_trees = best_i
    accuracies = []
    ratios = []

    for i in range(50, 80):
        flag = True
        i = i/100
        ratios.append(i)
        ratio = i
        acc = 0
        for j in range(3):
            acc += main()
            flag = False
        acc /= 3
        accuracies.append(acc)

    plt.plot(ratios, accuracies)
    plt.xlabel("Ratio of data")
    plt.ylabel("Accuracy (in %)")
    plt.title("Accuracy vs Ratio of data")
    plt.show()

if __name__ == "__main__":
    run_main()
