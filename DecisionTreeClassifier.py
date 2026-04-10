from BinaryTree import BinaryTree
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, data, target, threshold=0.0, features=None):
        self.threshold = threshold
        self.target = target
        if features is None:
            self.add_features(data)
        else:
            self.features = features
        data = data[self.features + [self.target]]
        self.root = self.build_tree(data)

    def add_features(self, data):
        temp_features = data.select_dtypes(include="number").columns.tolist()
        if self.target in temp_features:
            temp_features.remove(self.target)
        self.features = temp_features

    def gini_impurity(self, data):
        # get all the possible values at the label
        labels = data[self.target]

        # Get the probability of each value (normalize = True)
        probs = labels.value_counts(normalize=True)

        # Gini impurity formula = 1 - ∑i->n (p_i^2)
        return 1 - np.sum(probs.values ** 2)

    def gini_from_counts(self, counts, size):
        # Gini impurity formula = 1 - ∑i->n (p_i^2)
        the_sum = 0
        for count in counts.values():
            the_sum += ((count / size) ** 2)
        return 1 - the_sum

    def find_best_split(self, data, features, m_try):
        features_subset = list(np.random.permutation(features)[:m_try])

        best_feature = None
        best_threshold = None
        best_score = float('inf')

        for feature in features_subset:
            # Get the indices that map to a sorted array
            sorted_indices = np.argsort(data[feature].values, kind="mergesort")

            # get the sorted array for all features as well as the target feature (called labels)
            feature_values = data[feature].values[sorted_indices]
            labels = data[self.target].values[sorted_indices]
            n = len(data)

            # Get all unique labels from all labels in dataset
            unique_labels, counts = np.unique(labels, return_counts=True)

            left_counts = {}
            right_counts = {}

            # count labels in left and right. Initially left has nothing and right has all
            i = 0
            for label in unique_labels:
                left_counts[label] = 0
                right_counts[label] = counts[i]
                i += 1

            # At this point we have sorted the data by the feature in features_subset
            # and have the respective label as well
            for i in range(1, n):

                # With every increment in i, we get the label from all labels,
                # add it to the left and remove from the right
                label = labels[i - 1]
                left_counts[label] += 1
                right_counts[label] -= 1

                # If the value at this i is the same as the previous one, nothing new can be gathered so move ahead
                if feature_values[i] == feature_values[i - 1]:
                    continue

                # Required for gini impurity calculation
                left_size = i
                right_size = n - i

                # Find gini impurity score for left and right
                left_gini = self.gini_from_counts(left_counts, left_size)
                right_gini = self.gini_from_counts(right_counts, right_size)
                left_score = (left_size / n) * left_gini
                right_score = (right_size / n) * right_gini

                # Calculate final score (we want to minimise score)
                score = left_score + right_score
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = (feature_values[i] + feature_values[i - 1]) / 2

        return best_feature, best_threshold

    def build_tree(self, data):
        tree = BinaryTree()

        if len(data) == 0:
            return None

        # If only 1 type of node is left (the target node)
        if len(data[self.target].unique()) == 1:
            tree.set_data(data[self.target].iloc[0])
            return tree

        # Edge case where data is too small to properly fit a decision tree
        if len(data) < 2:
            tree.set_data(data[self.target].mode().iloc[0])
            return tree

        # Gini stop condition
        if self.gini_impurity(data) <= self.threshold:
            tree.set_data(data[self.target].mode().iloc[0])
            return tree

        # m_try = sqrt(num_features) and m_try > 0
        num_features = len(self.features)
        m_try = max(1, int(num_features ** 0.5))

        feature, threshold = self.find_best_split(data, self.features, m_try)

        # Edge case where no valid split was found
        if feature is None:
            tree.set_data(data[self.target].mode().iloc[0])
            return tree

        tree.set_feature(feature)
        tree.set_threshold(threshold)

        # left node is whatever is less than or equal to the thresold we calculated
        left_data = data[data[feature] <= threshold]
        # right node is whatever is more than the threshold
        right_data = data[data[feature] > threshold]

        # If no data, just make a tree with the mode (most frequent value)
        if len(left_data) == 0 or len(right_data) == 0:
            tree.set_data(data[self.target].mode().iloc[0])
            return tree

        left_tree = self.build_tree(left_data)
        right_tree = self.build_tree(right_data)

        tree.attach_left(left_tree)
        tree.attach_right(right_tree)

        return tree

    def predict_one(self, node, row):
        if node.get_data() is not None:
            return node.get_data()

        if row[node.get_feature()] <= node.get_threshold():
            return self.predict_one(node.get_left(), row)
        else:
            return self.predict_one(node.get_right(), row)
