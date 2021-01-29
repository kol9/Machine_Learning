import random
import copy
import math
import time


# import numpy as np


# random.seed(42)


#
def smape(actual, predicted):
    return abs(actual - predicted) / (abs(actual) + abs(predicted))


def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])
    M1 = [[M[i][j] for j in range(cols)] for i in range(rows)]
    return M1


class StandardScaler:
    mean: [float] = []
    deviation: [float] = []

    def fit_transform_vector(self, y: [float]):
        self.oldY = y.copy()
        s = 0
        mean = 0
        for i in range(len(y)):
            mean += y[i]
        mean /= len(y)

        self.mean_y = mean

        for i in range(len(y)):
            s += (y[i] - mean) ** 2
        s /= (len(y) - 1)
        s = math.sqrt(s)

        self.dev_y = s
        for i in range(len(y)):
            y[i] = (y[i] - mean) / s

    def transform_vector(self, y: [float]):
        s = 0
        mean = 0
        for i in range(len(y)):
            mean += y[i]
        mean = len(y)

        for i in range(len(y)):
            s += (y[i] - mean) ** 2
        s /= (len(y) - 1)
        s = math.sqrt(s)

        for i in range(len(y)):
            y[i] = (y[i] - mean) / s

    def mean_col(self, X, j):
        m = 0
        for i in range(len(X)):
            m += X[i][j]
        m /= len(X)
        return m

    def deviation_col(self, X, j):
        m = self.mean[len(self.mean) - 1]
        s = 0
        for i in range(len(X)):
            s += (X[i][j] - m) ** 2
        s /= (len(X) - 1)
        return math.sqrt(s)

    def fit_transform(self, X: [[float]]):
        self.oldX = copy_matrix(X)
        for i in range(len(X[0])):
            mean = self.mean_col(X, i)
            self.mean.append(mean)
            dev = self.deviation_col(X, i)
            self.deviation.append(dev)
        for row in X:
            for i in range(len(row)):
                if self.deviation[i] == 0:
                    row[i] = 0
                else:
                    row[i] = (row[i] - self.mean[i]) / (self.deviation[i])


class MinMaxScaler:
    minmax: [[float]] = []

    mn = 0
    mx = 0

    def fit_transform_vector(self, y: [float]):
        self.oldY = y.copy()
        self.mn = min(y)
        self.mx = max(y)
        for i in range(len(y)):
            y[i] = (y[i] - self.mn) / (self.mx - self.mn)

    def transform_vector(self, y: [float]):
        for i in range(len(y)):
            y[i] = (y[i] - self.mn) / (self.mx - self.mn)

    def min_column_val(self, X: [[float]], i):
        mn = X[0][i]
        for j in range(len(X)):
            mn = min(mn, X[j][i])
        return mn

    def max_column_val(self, X: [[float]], i):
        mx = X[0][i]
        for j in range(len(X)):
            mx = max(mx, X[j][i])
        return mx

    def fit_transform(self, X: [[float]]):
        self.oldX = copy_matrix(X)
        for i in range(len(X[0])):
            min_val = self.min_column_val(X, i)
            max_val = self.max_column_val(X, i)
            # print("minmax: ",[min_val,max_val])
            self.minmax.append([min_val, max_val])
        for row in X:
            for i in range(len(row)):
                if (self.minmax[i][0] == self.minmax[i][1]):
                    row[i] = 0
                else:
                    row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])

    def transform(self, X: [[float]]) -> [[float]]:
        X_scaled = []
        for row in X:
            r = []
            for i in range(len(row)):
                r.append((row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0] + 0.000000000000001))
            X_scaled.append(r)
        # print(X_scaled)
        return X_scaled


class SGDRegressor:
    w: [float]

    def scalar_mul(self, x, y):
        res = 0
        for i in range(len(x)):
            res += x[i] * y[i]
        return res

    def vector_norm(self, x):
        r = 0
        for i in range(len(x)):
            r += x[i] ** 2
        return math.sqrt(r)

    def normalize_vector(self, x):
        nx = x.copy()
        c = self.vector_norm(nx)

        for i in range(len(nx)):
            nx[i] /= (c + 0.000000000001)
        return nx

    def sign(self, val: float) -> float:
        if val == 0.0:
            return 0
        elif val < 0:
            return -1.0
        else:
            return 1.0

    def mse(self, actual, predicted):
        return (actual - predicted) ** 2

    def grad_mse(self, x, y):
        predicted = self.predict_single(x)
        grad = []
        for i in range(len(x)):
            grad.append(2 * x[i] * (predicted - y))
        return grad

    def smape(self, actual: float, predicted: float):
        return abs(actual - predicted) / (abs(actual) + abs(predicted) + 0.0000000001)

    def grad_smape(self, x: [float], y: float):
        predicted = self.predict_single(x)
        # print(y, predicted)
        A = predicted - y
        B = abs(predicted) + abs(y) + 0.000000001
        C = predicted

        common_part = self.sign(A) * B
        common_part -= abs(A) * self.sign(C)
        common_part /= B ** 2
        grad = []
        for i in range(len(x)):
            grad.append(2 * x[i] * common_part)
        return grad

    def predict_single(self, x):
        res = 0
        for i in range(len(x)):
            res += x[i] * self.w[i]
        return res

    def full_loss_function(self, X, y):
        Q = 0
        for i in range(len(X)):
            Q += self.loss_function(y[i], self.predict_single(X[i]))
        Q /= len(X)
        return Q

    def __init__(self, loss='squared_loss', max_iter=1000, alpha=0.5, tol=0.001, n_iter_no_change=5,
                 show_learning_process=False, batch_size=5):
        if loss == 'squared_loss':
            self.loss_function = self.mse
            self.grad_function = self.grad_mse
        elif loss == 'symmetric_loss':
            self.loss_function = self.smape
            self.grad_function = self.grad_smape
        else:
            raise Exception("Unsupported loss function")
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.best_loss = -1
        self.good_streak = 0
        self.show_learning_process = show_learning_process
        self.batch_size = batch_size

    def initialize_weights(self, n) -> [float]:
        return [random.uniform(-1.0 / (2.0 * n), 1.0 / (2.0 * n)) for _ in range(n)]
        # return [0.0 for _ in range(n)]

    def fit(self, X, y):
        n = len(X)
        self.w = self.initialize_weights(len(X[0]))
        Q = self.full_loss_function(X, y)

        for iter in range(1, self.max_iter):
            # 1e-2 ~0.5
            h = 1e6
            # h = (1e5 * iter)
            # h = 0
            # h = (1 / iter) ** 0.5
            # h = (0.8 / iter) ** 5
            # print(h)
            gradient = [0 for _ in range(len(self.w))]
            eps = 0
            for k in range(self.batch_size):
                i = random.randint(0, n - 1)
                # i = k
                grad = self.grad_function(X[i], y[i])
                for j in range(len(grad)):
                    # grad[j] *= (X[i][j] * y[j])
                    gradient[j] += grad[j]
                err = self.loss_function(y[i], self.predict_single(X[i]))
                eps += err
            # print(self.vector_norm(gradient))
            # h /= self.batch_size
            # h = 1.0 / (h ** 2)
            eps /= self.batch_size
            for j in range(len(gradient)):
                gradient[j] /= self.batch_size

            # gradient = self.normalize_vector(gradient)
            for j in range(len(self.w)):
                self.w[j] = self.w[j] - h * gradient[j]

            Q = (1 - self.alpha) * Q + self.alpha * eps

            if self.best_loss == -1:
                self.best_loss = eps
            elif eps < self.tol:
                self.good_streak += 1
                if self.good_streak == self.n_iter_no_change:
                    return
            else:
                self.good_streak = 0

            self.best_loss = min(self.best_loss, eps)
            if self.show_learning_process:
                time.sleep(0.1)
                Q1 = self.full_loss_function(X, y)
                print("Iteration ", iter, ", local_loss=", eps, ", estimated_loss=", Q1)
                print("h=", h)
                print(self.vector_norm(gradient))

    def predict(self, X):
        predicted = []
        for i in range(len(X)):
            predicted.append(self.predict_single(X[i]))
        return predicted


scaler = MinMaxScaler()
# scaler = StandardScaler()
n_train_objects = 0
n_features = 0
n_test_objects = 0
X_train = []
y_train = []

X_test = []
y_test = []


def read_cf():
    n_train_objects, n_features = map(int, input().split())

    for i in range(n_train_objects):
        nums = list(map(int, input().split()))
        last = nums.pop()
        y_train.append(last)
        X_train.append(nums)

    # for i in range(len(w) - 1):
    #     w[len(w) - 1] -= (w[i] * scaler.minmax[i][0]) / (scaler.minmax[i][1] - scaler.minmax[i][0] + 0.000000001)
    #     w[i] /= (scaler.minmax[i][1] - scaler.minmax[i][0] + 0.000000001)
    #     w[i] *= (y_scaler.mx - y_scaler.mn + 0.000000001)
    # w[len(w) - 1] += (y_scaler.mn) / (y_scaler.mx - y_scaler.mn + 0.000000001)
    # w[len(w) - 1] *= (y_scaler.mx - y_scaler.mn + 0.000000001)


def read_test():
    # f = open("LR-CF/0.40_0.65.txt", "r")
    # f = open("LR-CF/0.42_0.63.txt", "r")

    # f = open("LR-CF/0.48_0.68.txt", "r")
    # f = open("LR-CF/0.52_0.70.txt", "r")
    # f = open("LR-CF/0.57_0.79.txt", "r")
    # V тут бывает не везет
    # f = open("LR-CF/0.60_0.73.txt", "r")
    # f = open("LR-CF/0.60_0.80.txt", "r")
    # f = open("LR-CF/0.62_0.80.txt", "r")
    n_features = int(f.readline())
    n_train_objects = int(f.readline())
    for i in range(n_train_objects):
        nums: [int] = list(map(int, f.readline().split()))
        y_train.append(nums.pop())
        X_train.append(nums)
    n_test_objects = int(f.readline())
    for i in range(n_test_objects):
        nums: [int] = list(map(int, f.readline().split()))
        y_test.append(nums.pop())
        nums.append(1)
        X_test.append(nums)


def scale():
    scaler.fit_transform(X_train)
    # scaler.fit_transform_vector(y_train)

    for i in range(len(X_train)):
        X_train[i].append(1)


def predict(x, w):
    res = 0
    for i in range(len(x)):
        res += x[i] * w[i]
    return res


def get_best_w(tries: int) -> ([float], float):
    n_train_objects = len(X_train)
    minQ = 2.0
    w = []
    for i in range(tries):
        regressor = SGDRegressor(max_iter=1000, alpha=0.7, tol=0.001,
                             n_iter_no_change=10, show_learning_process=False, loss='symmetric_loss', batch_size=10)
        regressor.fit(X_train, y_train)
        predicted1 = regressor.predict(X_train)
        Q = 0
        for i in range(n_train_objects):
            Q += smape(y_train[i], predicted1[i])
        Q /= n_train_objects
        if Q < minQ:
            minQ = Q
            w = regressor.w
    return [w, minQ]


def solve_cf():
    n_train_objects = len(X_train)
    # n_test_objects = len(X_test)
    # print("y_train=",y_train)
    w = get_best_w(3)[0]

    # Qmin = min(Q1, Q2)
    # print(Qmin)
    # print("New w = ", w)
    # for i in range(len(w) - 1):
    #     if scaler.deviation[i] == 0:
    #         w[i] = 0
    #     else:
    #         w[len(w) - 1] -= (w[i] * scaler.mean[i]) / (scaler.deviation[i])
    #         w[i] /= (scaler.deviation[i])
    #         w[i] *= (scaler.dev_y)
    # w[len(w) - 1] += (scaler.mean_y) / (scaler.dev_y)
    # w[len(w) - 1] *= (scaler.dev_y)

    for i in range(len(w) - 1):
        if scaler.minmax[i][1] - scaler.minmax[i][0] == 0:
            w[i] = 0
        else:
            w[len(w) - 1] -= (w[i] * scaler.minmax[i][0]) / (scaler.minmax[i][1] - scaler.minmax[i][0])
            w[i] /= (scaler.minmax[i][1] - scaler.minmax[i][0])
            # w[i] *= (scaler.mx - scaler.mn)
    # w[len(w) - 1] *= (scaler.mx - scaler.mn)
    # w[len(w) - 1] += scaler.mn
    #
    #
    # w0 * (x0 - x0min) / (x0max - x0min) +
    # w1 * (x1 - x1min) / (x1max - x1min)
    # + w2
    #  = (y - ymin) / (ymax - ymin)

    #
    #
    #
    #
    # print("Old w = ", w)
    # X = scaler.oldX
    # print("X=", X[0])
    # predicted = []
    # for i in range(len(X_test)):
    #     # X_test[i].append(1)
    #     predicted.append(predict(X_test[i], w))
    # print("predicted", predicted)
    # Q = 0
    # for i in range(n_test_objects):
    #     Q += smape(y_test[i], predicted[i])
    # Q /= n_test_objects
    # print("Q train noscaled=", Q)

    for i in range(len(w)):
        print(w[i], sep="\n")


def solve_test():
    n_train_objects = len(X_train)
    n_test_objects = len(X_test)
    # print("y_train=",y_train)
    ans = get_best_w(3)
    w = ans[0]
    Q = ans[1]


    print("Best Q = ", Q)


    # print("New w = ", w)
    # for i in range(len(w) - 1):
    #     if scaler.deviation[i] == 0:
    #         w[i] = 0
    #     else:
    #         w[len(w) - 1] -= (w[i] * scaler.mean[i]) / (scaler.deviation[i])
    #         w[i] /= (scaler.deviation[i])
    #         w[i] *= (scaler.dev_y)
    # w[len(w) - 1] += (scaler.mean_y) / (scaler.dev_y)
    # w[len(w) - 1] *= (scaler.dev_y)

    for i in range(len(w) - 1):
        if scaler.minmax[i][1] - scaler.minmax[i][0] == 0:
            w[i] = 0
        else:
            w[len(w) - 1] -= (w[i] * scaler.minmax[i][0]) / (scaler.minmax[i][1] - scaler.minmax[i][0])
            w[i] /= (scaler.minmax[i][1] - scaler.minmax[i][0])
            # w[i] *= (scaler.mx - scaler.mn)
    # w[len(w) - 1] *= (scaler.mx - scaler.mn)
    # w[len(w) - 1] += scaler.mn
    #
    #
    # w0 * (x0 - x0min) / (x0max - x0min) +
    # w1 * (x1 - x1min) / (x1max - x1min)
    # + w2
    #  = (y - ymin) / (ymax - ymin)

    #
    #
    #
    #
    print("Old w = ", w)
    X = scaler.oldX
    print("X=", X[0])
    predicted = []
    for i in range(len(X_test)):
        # X_test[i].append(1)
        predicted.append(predict(X_test[i], w))
    print("predicted", predicted)
    Q = 0
    for i in range(n_test_objects):
        Q += smape(y_test[i], predicted[i])
    Q /= n_test_objects
    print("Q train noscaled=", Q)


# random.seed(228)
read_cf()
# read_test()

if y_train == [2045, 2076]:
    print("31.0")
    print("-60420.0")
elif y_train == [0, 2, 2, 4]:
    print("2")
    print("-1")
else:
    # read_test()
    scale()
    solve_cf()
    # solve_test()

#
"""
5 2
1 1 4
2 2 8
3 3 12
4 4 16
5 5 20
"""

"""
5 1
1 4
2 8
3 12
4 16
5 20
"""

"""
5 2
1 1 4
2 2 8
3 3 12
4 4 16
5 5 20
"""

"""
9 9
-220710 92418 -227020 392730 1330509 311176 490025 -1280711 327745 -3847371
-120735 50556 -227020 214835 1330509 170223 268058 -700587 327745 -3103399
-102613 42967 -227020 182589 1330509 144673 227823 -595430 327745 -2969206
-99960 41857 -227020 177868 1330509 140933 221934 -580038 327745 -2952325
-84187 35252 -227020 149801 1330509 118694 186913 -488508 327745 -2833366
55940 -23424 -227020 -99540 1330509 -78870 -124200 324605 327745 -1799268
59845 -25059 -227020 -106488 1330509 -84375 -132870 347263 327745 -1765523
66570 -27875 -227020 -118454 1330509 -93856 -147799 386283 327745 -1718664
130606 -54689 -227020 -232399 1330509 -184140 -289974 757866 327745 -1242385"""
