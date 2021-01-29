import math


def mean(arr):
    n = len(arr)
    sum = 0
    for i in range(n):
        sum += arr[i]
    sum /= n
    return sum







N = int(input())
F = []
S = []
for i in range(N):
    f, s = map(int,input().split())
    F.append(f)
    S.append(s)


mean_F = mean(F)
mean_S = mean(S)

S1 = 0
for i in range(N):
    S1 += (F[i] - mean_F) * (S[i] - mean_S)

S2 = 0
for i in range(N):
    S2 += (F[i] - mean_F) ** 2

S3 = 0
for i in range(N):
    S3 += (S[i] - mean_S) ** 2



sq = math.sqrt((S2) * (S3))
if sq == 0:
    print(0)
else:
    print(S1 / sq)