class BinaryTree:
    def __init__(self, data=None, threshold=None, feature=None):
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.right = None
        self.left = None
        self.parent = None

    def set_data(self, data):
        self.data = data

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_feature(self, feature):
        self.feature = feature

    def set_left(self, tree):
        self.left = tree

    def set_right(self, tree):
        self.right = tree

    def set_parent(self, tree):
        self.parent = tree

    def get_data(self):
        return self.data

    def get_threshold(self):
        return self.threshold

    def get_feature(self):
        return self.feature

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_parent(self):
        return self.parent

    def attach_left(self, tree):
        if tree is None:
            return
        elif self.left is not None or tree.get_parent() is not None:
            raise ValueError("Unable to attach left")
        else:
            tree.set_parent(self)
            self.set_left(tree)

    def attach_right(self, tree):
        if tree is None:
            return
        elif self.right is not None or tree.get_parent() is not None:
            raise ValueError("Unable to attach right")
        else:
            tree.set_parent(self)
            self.set_right(tree)

    def root(self):
        if self.get_parent() is None:
            return self

        node = self
        while node.get_parent() is not None:
            node = node.get_parent()
        return node
