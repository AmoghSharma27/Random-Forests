from BinaryTree import BinaryTree
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, data, threshold=0.0, target="Type 1", features=None):
        if features is None:
            features = data.select_dtypes(include="number").columns.tolist()
            if target in features:
                features.remove(target)
        self.threshold = threshold
        self.target = target
        self.features = features

        data = data[self.features + [self.target]]

        self.root = self.build_tree(data)

    def add_features(self, data):
        features = []
        # Return all columns that are not the target
        for column in data.columns:
            if column != self.target:
                features.append(column)
        return features

    def gini_impurity(self, data):
        # get all the possible values at the label
        labels = data[self.target]

        # Get the probability of each value (normalize = True)
        probs = labels.value_counts(normalize=True)

        # Gini impurity formula = 1 - ∑i->n (p_i^2)
        impurity = 0
        for p in probs:
            impurity += p * p

        return 1 - impurity

    def find_best_split(self, data, features, m_try):
        feature_subset = list(np.random.choice(features, m_try, replace=False))

        best_feature = None
        best_threshold = None
        best_score = float('inf')

        for feature in feature_subset:
            unique_values = sorted(data[feature].unique())
            thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

            for threshold in thresholds:
                left = data[data[feature] <= threshold]
                right = data[data[feature] > threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                left_weight = (len(left) / len(data))
                right_weight = (len(right) / len(data))

                left_weight *= self.gini_impurity(left)
                right_weight *= self.gini_impurity(right)

                score = left_weight + right_weight

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, data):
        tree = BinaryTree()

        # If only 1 type of node is left (the target node)
        if len(data[self.target].unique()) == 1:
            tree.set_data(data[self.target].iloc[0])
            return tree

        # Edge case where data is too small to properly fit a decision tree
        if len(data) < 2:
            tree.set_data(data[self.target].mode()[0])
            return tree

        # Gini stop condition
        if self.gini_impurity(data) <= self.threshold:
            tree.set_data(data[self.target].mode()[0])
            return tree

        # m_try = sqrt(num_features) and m_try > 0
        num_features = len(self.features)
        m_try = max(1, int(num_features ** 0.5))

        feature, threshold = self.find_best_split(data, self.features, m_try)

        # Edge case where no valid split was found
        if feature is None:
            tree.set_data(data[self.target].mode()[0])
            return tree

        tree.set_feature(feature)
        tree.set_threshold(threshold)

        # left node is whatever is less than or equal to the thresold we calculated
        left_tree = self.build_tree(data[data[feature] <= threshold])
        # right node is whatever is more than the threshold
        right_tree = self.build_tree(data[data[feature] > threshold])

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
