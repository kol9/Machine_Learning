def to_bits(x: int, l: int):
    c = x
    res = []
    while c > 0:
        d, m = divmod(c, 2)
        res.append(m)
        c = d

    while len(res) < l:
        res.append(0)
    return res


D = int(input())
t = dict()

def remove_zeros(t, D):
    # new_t = dict()
    for k in range(1 << D):
        a, r = t[k]
        if r == 0:
            t.pop(k)


def remove_ones(t, D):
    # new_t = dict()
    for k in range(1 << D):
        a, r = t[k]
        if r == 1:
            t.pop(k)



def solve_CNF(t):
    remove_ones(t, D)
    print(2)
    print(len(t), 1)
    # 1st layer
    for k in t:
        a, f = t[k]
        count_pos = 0
        for i in range(len(a)):
            if a[i] == 0:
                print(100, end=" ")
            else:
                print(-1, end=" ")
                count_pos += 1
        print(count_pos - 0.5)
    # 2nd layer
    for i in range(len(t)):
        print(1, end=" ")
    print(-(len(t) - 0.5))


def solve_DNF(t):
    remove_zeros(t, D)
    print(2)
    print(len(t), 1)
    for k in t:
        a, f = t[k]
        count_pos = 0
        for i in range(len(a)):
            if a[i] == 0:
                print(-100, end=" ")
            else:
                count_pos += 1
                print(1, end=" ")
        print(-(count_pos - 0.5))
    for i in range(len(t)):
        print(1, end=" ")
    print(-0.5)



def solve_const_zero():
    print(1)
    print(1)
    for i in range(D):
        print(1,end=" ")
    print(-100)

def solve_const_one():
    print(1)
    print(1)
    for i in range(D):
        print(1, end=" ")
    print(100)





cnt = dict()
for mask in range(1 << D):
    r = int(input())

    if r not in cnt:
        cnt[r] = 0
    cnt[r] += 1
    bits = to_bits(mask, D)
    t[mask] = (bits, r)


if 0 in cnt and 1 in cnt:
    if cnt[0] < cnt[1]:
        solve_CNF(t)
    else:
        solve_DNF(t)
else:
    if 0 not in cnt:
        solve_const_one()
    else:
        solve_const_zero()
