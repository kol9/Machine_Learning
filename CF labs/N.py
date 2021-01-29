import math


def calc(nums):
    if len(nums) == 0:
        return 0
    ans = 0
    nums = list(reversed(sorted(nums)))
    n = len(nums)
    prefix_sum = [0 for _ in range(n)]

    prefix_sum[0] = nums[0]
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i - 1] + nums[i]

    def get_sum(i, j):
        if i == 0:
            return prefix_sum[j]
        return prefix_sum[j] - prefix_sum[i - 1]

    for i in range(n - 1):
        ans += (n - i - 1) * nums[i] - get_sum(i + 1, n - 1)
    return ans


K = int(input())
N = int(input())
X = []
mapper = dict()

for i in range(N):
    x, y = map(int, input().split())
    if y not in mapper:
        mapper[y] = []
    mapper[y].append(x)
    X.append(x)

inclass = 0
for i in range(K):
    j = i + 1
    if j in mapper:
        res = calc(mapper[j])
        inclass += res
inclass *= 2

total = calc(X)
total *= 2

print(inclass)
print(total - inclass)
