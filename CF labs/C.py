import math


def distance_function(x: [float], y: [float], distance: str = 'euclidean') -> float:
    # assert (len(x) == len(y))

    if distance == 'manhattan':
        dist = 0
        for i in range(len(x)):
            dist += abs(x[i] - y[i])
        return dist
    elif distance == 'euclidean':
        dist = 0
        for i in range(len(x)):
            dist += abs(x[i] - y[i]) ** 2
        return dist ** 0.5
    elif distance == 'chebyshev':
        dist = 0
        for i in range(len(x)):
            dist = max(abs(x[i] - y[i]), dist)
        return dist
    else:
        raise Exception("Unsupported distance function")


def kernel_function(u: float, kernel: str = 'gaussian') -> float:
    try:
        if kernel == 'uniform':
            if (abs(u) < 1):
                return 0.5
            else:
                return 0
        elif kernel == 'triangular':
            if (abs(u) < 1):
                return 1 - abs(u)
            else:
                return 0
        elif kernel == 'epanechnikov':
            if (abs(u) < 1):
                return 0.75 * (1 - u ** 2)
            else:
                return 0
        elif kernel == 'quartic':
            if (abs(u) < 1):
                return (15 / 16) * (1 - u ** 2) ** 2
            else:
                return 0
        elif kernel == 'triweight':
            if (abs(u) < 1):
                return (35 / 32) * (1 - u ** 2) ** 3
            else:
                return 0
        elif kernel == 'tricube':
            if (abs(u) < 1):
                return (70 / 81) * (1 - abs(u) ** 3) ** 3
            else:
                return 0
        elif kernel == 'gaussian':
            return (1 / (2 * math.pi) ** 0.5) * math.e ** (-0.5 * u ** 2)
        elif kernel == 'cosine':
            if (abs(u) < 1):
                return (math.pi / 4) * math.cos((math.pi / 2) * u)
            else:
                return 0
        elif kernel == 'logistic':
            return 1 / (math.e ** u + 2 + math.e ** (-u))
        elif kernel == 'sigmoid':
            return (2 / math.pi) * 1 / ((math.e ** u) + (math.e ** (-u)))
        else:
            raise Exception("Unsupported kernel function")
    except:
        return 0


def nonParamRegression(x_test: [float], X_train: [[float]], y_train: [float],
                       distance: str,
                       kernel: str,
                       window_type: str,
                       window_val: float):
    dists = [distance_function(X_train[i], x_test, distance=distance) for i in range(len(X_train))]
    dists.sort()

    def mean(arr: [float]):
        sum = 0
        for i in range(len(arr)):
            sum += arr[i]
        sum /= len(arr)
        return sum

    def w_i(i: int) -> float:
        if window_type == 'fixed':
            return kernel_function(
                distance_function(x_test, X_train[i], distance=distance) / window_val
                , kernel=kernel)
        else:
            new_window_val = min(window_val, len(dists) - 1)
            return kernel_function(
                distance_function(x_test, X_train[i], distance=distance) / dists[int(new_window_val)]
                , kernel=kernel)

    up = 0
    down = 0

    if window_type == 'fixed' and window_val == 0:
        good = []
        for i in range(len(y_train)):
            if distance_function(X_train[i], x_test, distance=distance) == 0:
                good.append(y_train[i])
        if len(good) != 0:
            return mean(good)
        else:
            return mean(y_train)

    for i in range(len(y_train)):
        try:
            down += w_i(i)
            up += y_train[i] * w_i(i)
        except:
            pass
    try:
        ans = up / down
    except:
        good = []
        ans = mean(y_train)
        for i in range(len(y_train)):
            if distance_function(X_train[i], x_test, distance=distance) == 0:
                good.append(y_train[i])
        if len(good) != 0:
            ans = mean(good)
    return ans


N, M = map(int, input().split())

ds = []

y_train = []
X_train = []
for i in range(N):
    nums = list(map(int, input().split()))
    ds.append(nums)
    y_train.append(nums.pop())
    X_train.append(nums)
x_test = list(map(int, input().split()))

distance_param = input()
kernel_param = input()
window_type = input()
kh = 0
if window_type == 'fixed':
    kh = int(input())
else:
    kh = int(input())

print(nonParamRegression(x_test, X_train, y_train,
                         distance=distance_param,
                         kernel=kernel_param,
                         window_type=window_type,
                         window_val=kh))