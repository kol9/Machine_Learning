import math
import random


class DecisionTreeClassifier:
    class Node:
        delim = None
        sep_feature = None
        cls = None
        left = None
        right = None
        id = 0

        def __init__(self, left, right, sep_feature, delim, cls):
            self.left = left
            self.right = right
            self.sep_feature = sep_feature
            self.delim = delim
            self.cls = cls

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.X = None
        self.y = None
        self.n_nodes = 0

    def gini(self, classes):
        count = dict()
        for c in classes:
            if c not in count:
                count[c] = 0
            count[c] += 1
        sum = 0
        for c in count.keys():
            p = count[c] / len(classes)
            sum += p ** 2
        return 1 - sum

    def entropy(self, classes):
        count = dict()
        for c in classes:
            if c not in count:
                count[c] = 0
            count[c] += 1
        probs = []
        for c in count.keys():
            probs.append(count[c] / len(classes))
        sum = 0
        for p in probs:
            sum += p * math.log2(p)
        sum = -sum
        return sum

    def Igain(self, xs, feature, sep):
        if len(xs) == 0:
            while True:
                pass
        first_block_y = []
        second_block_y = []

        first_indices = []
        second_indices = []

        for i in xs:
            if self.X[i][feature] < sep:
                first_block_y.append(self.y[i])
                first_indices.append(i)
            else:
                second_block_y.append(self.y[i])
                second_indices.append(i)

        i_gain = self.gini(xs) - (len(first_block_y) / len(xs)) * self.gini(first_block_y) - \
                 (len(second_block_y) / len(xs)) * self.gini(second_block_y)

        return i_gain, first_indices, second_indices

    def fit(self, X, y):
        self.X = X
        self.y = y
        if len(X) > 3000:
            self.root = self._build_tree([i for i in range(0, len(self.X), 2)])
        else:
            self.root = self._build_tree([i for i in range(0, len(self.X))])
        pass

    def _build_tree(self, xs, depth=0) -> Node:
        if depth == self.max_depth or len(xs) <= 2:
            classes_cnt = dict()
            for i in xs:
                if self.y[i] not in classes_cnt:
                    classes_cnt[self.y[i]] = 0
                classes_cnt[self.y[i]] += 1
            popular = 0
            cnt = -1
            for c in classes_cnt.keys():
                if classes_cnt[c] > cnt:
                    popular = c
                    cnt = classes_cnt[c]
            self.n_nodes += 1
            node = self.Node(None, None, None, None, popular)
            return node

        best_i_gain = float('-inf')
        first_indices = []
        second_indices = []
        delimeter = 0
        best_feature = 0

        for feature in range(len(self.X[0])):
            values = []
            mn = float('inf')
            mx = float('-inf')
            for i in range(0, len(xs), 2):
                mn = min(mn, self.X[xs[i]][feature])
                mx = max(mx, self.X[xs[i]][feature])
                # values.append(self.X[i][feature])
            # values.sort()
            # mn = values[0]
            # mx = values[len(values) - 1]
            step = (mx - mn) / 12
            for i in range(1, 12):
                # for i in range(len(values) - 1):
                sep = mn + step * i
                # sep = (values[i] + values[i + 1]) / 2
                i_gain, b1, b2 = self.Igain(xs, feature, sep)
                if i_gain > best_i_gain:
                    best_feature = feature
                    best_i_gain = i_gain
                    first_indices = b1
                    second_indices = b2
                    delimeter = sep

        left = self._build_tree(first_indices, depth + 1)
        right = self._build_tree(second_indices, depth + 1)

        self.n_nodes += 1
        node = self.Node(left, right, best_feature, delimeter, None)
        return node

    def predict(self, X):
        res = []
        for x in X:
            res.append(self._traverse(self.root, x))
        return res

    def _traverse(self, node: Node, vect):
        if node.left is None or node.right is None:
            return node.cls
        if vect[node.sep_feature] < node.delim:
            return self._traverse(node.left, vect)
        else:
            return self._traverse(node.right, vect)

    def print_tree(self):

        self.print_mapper = dict()
        self.counter = 1
        self._print_traverse(self.root)

        total = set()

        cur = 2
        for i in self.print_mapper.keys():
            total.add(i)

        size = len(total)
        print(size)

        for i in range(1, size + 1):
            str = self.print_mapper[i]
            print(str)

    def _print_traverse(self, node: Node):

        node.id = self.counter
        self.counter += 1
        if node.left is None or node.right is None:
            self.print_mapper[node.id] = "C " + str(node.cls)
            return

        self._print_traverse(node.left)
        self._print_traverse(node.right)
        self.print_mapper[node.id] = "Q " + str(node.sep_feature + 1) + " " + \
                                     str(node.delim) + " " + str(node.left.id) + " " + str(node.right.id)


n_features, n_classes, max_depth = map(int, list(input().split()))
n_objects = int(input())

X_train = []
y_train = []
for i in range(n_objects):
    nums = list(map(int, list(input().split())))
    y_train.append(nums.pop())
    X_train.append(nums)

clf = DecisionTreeClassifier(max_depth=max_depth)

clf.fit(X_train, y_train)
clf.print_tree()
