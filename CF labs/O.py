K = int(input())
N = int(input())

X = []
Y = []
for i in range(N):
    f, s = map(int, input().split())
    X.append(f)
    Y.append(s)

"""
Var(X)      = M(X^2)      - (M(X))^2
Var(Y|X)    = M(Y^2|X)    - (M(Y|X))^2
M(Var(Y|X)) = M(M(Y^2|X)) - M(M(Y^2|X)^2)

Law of Total Variance:
Var(Y)=M[Var(Y|X)]+Var(M[Y|X])
M[Var(Y|X)] = Var(Y) - Var(M[Y|X])

Var(M[Y|X]) = M(M[Y|X] ^ 2) - (M(M[Y|X]))^2
Var(M[Y|X]) = M(M[Y|X] ^ 2) - (M(Y))^2

M[Var(Y|X)] = Var(Y) - M(M[Y|X] ^ 2) + (M(Y))^2
M[Var(Y|X)] = M(Y^2) - M(M[Y|X] ^ 2)
M[Var(Y|X)] = M(Y^2) - M(M[Y|X] ^ 2)
sum y^2 * p 
"""

f = 0
s = 0
for i in range(N):
    y = Y[i]
    f += y ** 2 / N


CNT = dict()
M = dict()

for i in range(N):
    x = X[i]
    if x not in CNT:
        CNT[x] = 0
    CNT[x] += 1
    if x not in M:
        M[x] = 0
    M[x] += Y[i] / N



print(CNT)
print(M)

for i in range(1, K + 1):
    if i in CNT:
        s += (N * M[i] * M[i]) / (CNT[i])


print(f - s)