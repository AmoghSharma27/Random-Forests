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

        print("Building tree!")
        self.root = self.build_tree(data)

    def add_features(self, data):
        features = []
        # Return all columns that are not the target
        for column in data.columns:
            if column != self.target:
                features.append(column)
        return features

    def gini_impurity(self, labels):
        # Get counts for each unique label in labels
        labels = np.asarray(labels, dtype=str)
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Calculate probability of each label (we don't care about specific label, just the probability of each label)
        probs = counts / len(labels)
        # Gini impurity formula = 1 - ∑i->n (p_i^2)
        return 1 - np.sum(probs ** 2)

    def find_best_split(self, data, features, m_try):
        sub_features = np.random.choice(features, m_try, replace=False)

        best_feature = None
        best_threshold = None
        best_score = float('inf')

        # For each feature in selected subset of all features
        for feature in sub_features:
            # faster sorting because it uses numpy instead of pandas due to overhead in pandas
            # But essentially the same as unique_values = sorted(data[feature].unique())
            # https://penandpants.com/2014/09/05/performance-of-pandas-series-vs-numpy-arrays/
            values = data[feature].to_numpy()
            labels = data[self.target].to_numpy()

            order = np.argsort(values, kind='mergesort')
            values, labels = values[order], labels[order]

            unique_values = np.unique(values)

            # Only 1 possible label, so just skip
            if len(unique_values) <= 1:
                continue

            for i in range(len(unique_values) - 1):
                # Take threshold as the midpoint between 2 successive values in the unique values in the feature
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                left_labels = labels[values <= threshold]
                right_labels = labels[values > threshold]

                if len(left_labels) == 0 or len(right_labels) == 0:
                    # continue if either side has no values since it is not a threshold at that point
                    continue

                # n1/n:
                left_weight = len(left_labels) / len(data)
                # n2/n:
                right_weight = len(right_labels) / len(data)

                # Gini_A(D) = ((n1/n)*Gini(D1) + (n2/n)*Gini(D2))
                left_score = (left_weight) * self.gini_impurity(left_labels)
                right_score = (right_weight) * self.gini_impurity(right_labels)

                Gini_score = left_score + right_score

                # We are trying to minimise the Gini impurity score
                if Gini_score < best_score:
                    best_score = Gini_score
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
        if self.gini_impurity(data[self.target]) <= self.threshold:
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
