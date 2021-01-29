N, M, K = map(int, input().split())

nums: [int] = list(map(int, input().split()))

mapper = {}
for (i, e) in enumerate(nums):
    if not (e in mapper):
        mapper[e] = []
    mapper[e].append(i)

ans = [[] for _ in range(0, K)]


ptr = 0
for k in mapper:
    for i in mapper[k]:
        ans[ptr].append(i + 1)
        ptr += 1
        ptr %= K
for i in range(K):
    print(len(ans[i]), end=" ")
    for j in ans[i]:
        print(j, end=" ")
    print()
