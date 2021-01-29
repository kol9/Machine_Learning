import math

def prob(x, c) -> [int, int]:
    if x not in one_more_map[c]:
        up = alpha
    else:
        up = one_more_map[c][x] + alpha
    down = 0
    down += count_class[c]
    down += alpha * 2
    return [up, down]


n_classes = int(input())
lam_c = list(map(int, input().split()))
alpha = int(input())
n_training_objects = int(input())
y_train = []
X_train = []
X_test = []

count_class = dict()
one_more_map = dict()

all_words = set()

for i in range(n_training_objects):
    tokens = input().split()
    y_train.append(int(tokens[0]))
    words = []
    cls = int(tokens[0])
    if cls not in count_class:
        count_class[cls] = 0
    count_class[cls] += 1
    for j in range(2, len(tokens)):
        word = tokens[j]
        all_words.add(word)
        words.append(word)
    X_train.append(words)
    words = list(set(words))
    for w in words:
        if cls not in one_more_map:
            one_more_map[cls] = dict()
        if w not in one_more_map[cls]:
            one_more_map[cls][w] = 0
        one_more_map[cls][w] += 1
n_test_objects = int(input())
for i in range(n_test_objects):
    tokens = input().split()
    words = []
    for j in range(1, len(tokens)):
        words.append(tokens[j])
    X_test.append(set(words))

ans = []

mem = []

for test in X_test:
    cl = []
    sum = 0
    c = 0
    for cls in range(1, n_classes + 1):
        c += 1
        if cls not in count_class:
            cl.append(0)
        else:
            pred = 0
            pred += math.log(lam_c[cls - 1] * count_class[cls])
            pred -= math.log(n_training_objects)
            for wrd in all_words:
                if wrd in test:
                    pr = prob(wrd, cls)
                    pred += math.log(pr[0])
                    pred -= math.log(pr[1])
                else:
                    pr = prob(wrd, cls)
                    pred += math.log(pr[1] - pr[0])
                    pred -= math.log(pr[1])
            cl.append(pred)
            sum += pred
        # pred = math.exp(pred)
    mem.append(sum / c)
    ans.append(cl)


def vector_norm(x):
    r = 0
    for i in range(len(x)):
        r += x[i]
    return r




def normalize_vector(x):
    nx = x.copy()
    c = vector_norm(nx)
    # print("c=", c)
    for i in range(len(nx)):
        if nx[i] != 0:
            nx[i] /= c
            # nx[i] = math.exp(nx[i])
        # nx[i] /= (c)
        pass
    return nx


# def normalize_vector2(x: [[float]]):
#     nx = x.copy()
#     c = vector_sum(nx)
#
#     # if c == 0:
#     #     return nx
#     for i in range(len(nx)):
#         nx[i][0] *= c[1]
#         nx[i][1] *= c[0]
#     return nx




for i in range(len(ans)):
    l = mem[i]
    # print("l=", l)
    for j in range(0, len(ans[i])):
        if (ans[i][j] != 0):
            ans[i][j] -= l
            ans[i][j] = math.exp(ans[i][j])
for i in range(len(ans)):
    ans[i] = normalize_vector(ans[i])

for i in range(len(ans)):
    for x in ans[i]:
        print(x, end=" ")
    print()

"""
2
1 1
1
3
1 3 give uslugi bug
1 3 go buy viagra
2 3 need buy milk
1
3 need buy cigarettes
"""

"""
2
1 1
1
5
1 3 a great game
2 4 the election was over
1 3 very clean match
1 5 a clean but forgettable game
2 5 it was a close election
1
4 a very close game
"""

"""
3
1 1 1
1
4
1 9 a b c d e f h i g
2 9 h u j k l m n o p
1 18 r s t f q r s t u v w x y z aa bb cc dd
3 27 ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt uu vv xx yy zz aaa bbb ccc ddd eee fff
5
2 a a
5 b a b a a
5 d a d c a
2 a c
1 c
"""

"""
3
1 1 1
1
1
1 2 ant emu
5
2 emu emu
5 emu dog fish dog fish
5 fish emu ant cat cat
2 emu cat
1 cat
"""
