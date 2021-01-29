import random

n_train_objects = int(input())
K = []
y_train = []
for i in range(n_train_objects):
    nums = list(map(int, input().split()))
    y_train.append(nums.pop())
    K.append(nums)
C = int(input())


class SVM:

    def __init__(self, y, max_iter=100, tol=0.00000000001, C=1.0, eps=0.00000000001):
        # self.n_objects=len(X)
        self.n_objects = len(y)
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.target = y
        self.eps = eps
        self.alphas = [0.0 for _ in range(len(y))]
        self.w = [0.0 for _ in range(len(y))]
        self.b = 0.0

    def kernel(self, i, j):
        return K[i][j]
        # return 0
        # TODO

    def u_x(self, i) -> float:
        res = 0.0
        for j in range(self.n_objects):
            res += self.target[j] * self.alphas[j] * self.kernel(i, j)
        res += self.b
        return res

    def fit(self):
        for step in range(1, self.max_iter + 1):
            for i2 in range(self.n_objects):
                y2 = self.target[i2]
                E2 = self.u_x(i2) - y2
                alpha2 = self.alphas[i2]
                if (y2 * E2 < -self.tol and alpha2 < self.C) or (
                        y2 * E2 > self.tol and alpha2 > 0):
                    while True:
                        i1 = random.randint(0, self.n_objects - 1)
                        if i1 != i2:
                            break
                    alpha1 = self.alphas[i1]

                    y1 = self.target[i1]

                    E1 = self.u_x(i1) - y1
                    s = y1 * y2
                    if y1 != y2:
                        L = max(0.0, alpha2 - alpha1)
                        H = min(self.C, self.C + alpha2 - alpha1)
                    else:
                        L = max(0.0, alpha2 + alpha1 - self.C)
                        H = min(self.C, alpha2 + alpha1)
                    if L == H:
                        continue
                    k11 = self.kernel(i1, i1)
                    k12 = self.kernel(i1, i2)
                    k22 = self.kernel(i2, i2)

                    eta = k11 + k22 - 2 * k12

                    if eta > 0:
                        a2 = alpha2 + y2 * (E1 - E2) / eta
                        if a2 < L:
                            a2 = L
                        elif a2 > H:
                            a2 = H
                    else:
                        continue
                    if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
                        continue
                    a1 = alpha1 + s * (alpha2 - a2)

                    b1 = -(E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12) + self.b
                    b2 = -(E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22) + self.b

                    if 0 < a1 < self.C:
                        self.b = b1
                    elif 0 < a2 < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    # self.w = self.w + y1 * (a1 - alpha1) * self.X[i1] + y2 * (a2 - alpha2) * self.X[i2]
                    self.alphas[i1] = a1
                    self.alphas[i2] = a2


clf = SVM(y_train, max_iter=10000, C=C)
clf.fit()

for e in clf.alphas:
    if e >= 0.0:
        print(e)
    else:
        print(0.0)
print(clf.b)
