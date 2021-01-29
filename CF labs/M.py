import math


def mean(arr):
    n = len(arr)
    sum = 0
    for i in range(n):
        sum += arr[i]
    sum /= n
    return sum


def get_rangs(arr):
    n = len(arr)
    srtd = sorted([v, i] for (i, v) in enumerate(arr))
    ans = [0 for i in range(n)]
    c = 0
    for i in range(1, n):
        if srtd[i][0] != srtd[i - 1][0]:
            c += 1
        ans[srtd[i][1]] = c
    return ans


N = int(input())
X = []
Y = []
for i in range(N):
    f, s = map(int, input().split())
    X.append(f)
    Y.append(s)

rangsx = get_rangs(X)
rangsy = get_rangs(Y)

s = 0

for i in range(N):
    s += (rangsx[i] - rangsy[i]) ** 2

print(1 - 6 * (s) / (N * (N - 1) * (N + 1)))


"""
14
28	21
30	25
36	29
40	31
30	32
46	34
56	35
54	38
60	39
56	41
60	42
68	44
70	46
76	50
"""
