class BinaryTree:
    def __init__(self, data=None, threshold=None, feature=None):
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.right = None
        self.left = None
        self.parent = None

    def is_empty(self):
        return self.data is None

    def make_root(self, data):
        if not self.is_empty():
            print("Root already exists")
        else:
            self.data = data

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

    def detach_left(self):
        if self.is_empty():
            return None
        retLeft = self.get_left()
        self.left = None
        if retLeft is not None:
            retLeft.set_parent(None)
        return retLeft

    def detach_right(self):
        if self.is_empty():
            return None
        retRight = self.get_right()
        self.right = None
        if retRight is not None:
            retRight.set_parent(None)
        return retRight

    def clear(self):
        self.left = self.right = self.parent = None
        self.data = None
        self.threshold = None

    def root(self):
        if self.get_parent() is None:
            return self

        node = self
        while node.get_parent() is not None:
            node = node.get_parent()
        return node

    def depth(self):
        left_depth = right_depth = 0
        if self.get_left():
            left_depth = self.left.depth()
        if self.get_right():
            right_depth = self.right.depth()

        return 1 + max(left_depth, right_depth)

    def nodes(self):
        left_nodes = right_nodes = 0

        if self.get_left():
            left_nodes = self.get_left().nodes()
        if self.get_right():
            right_nodes = self.get_right().nodes()

        return 1 + left_nodes + right_nodes

    def pretty_print(self):
        if self.get_left():
            self.get_left().pretty_print()
        print(str(self.get_data()) + "\t")
        if self.get_right():
            self.get_right().pretty_print()
