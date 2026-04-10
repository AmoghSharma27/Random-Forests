from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np


class RandomForests:
    def __init__(self, train_data, num_trees=10, target="Type 1", forest=None):
        self.num_trees = num_trees
        self.target = target
        self.train_data = train_data
        if forest is None:
            self.forest = self.train_trees(train_data)
        else:
            self.forest = forest

    def get_forest(self):
        return (self.forest, self.train_data, self.num_trees, self.target)

    # Make bootstrap samples of the training data
    def bootstrap(self, train_data):
        n = len(train_data)
        indices = np.random.choice(n, size=n, replace=True)
        return train_data.iloc[indices]

    # Train the number of trees as specified on the bootstrap samples and append them to a list
    def train_trees(self, train_data):
        forest = []
        print(f"Attempting to train trees with {self.num_trees} trees")
        for i in range(self.num_trees):
            # Create a decision tree and train it on th bootstrapped training data
            sample = self.bootstrap(train_data)
            curr_tree = DecisionTreeClassifier(sample, self.target)
            forest.append(curr_tree)

        return forest

    def predict(self, data_to_predict_with):
        # Variable to store the majority vote prediction
        majority_pred = None
        majority_votes = 0

        # Dictionary of predictions
        predictions = {}

        # Keep track of each prediction in a dictionary with the amount of times a tree makes that prediction as value.
        for tree in self.forest:
            prediction = tree.predict_one(tree.root, data_to_predict_with)
            if prediction not in predictions.keys():
                predictions[prediction] = 1

            else:
                predictions[prediction] += 1

        # Go through all the predictions and return the most voted one
        for prediction, votes in predictions.items():
            if votes > majority_votes:
                majority_votes = votes
                majority_pred = prediction

        return majority_pred
