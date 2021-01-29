import math

K = int(input())

CM = []
for i in range(K):
    line = list(map(int, input().split()))
    CM.append(line)


def f1_score(confusion_matrix: [[int]]):
    n = len(confusion_matrix)

    TP = [confusion_matrix[x][x] for x in range(n)]

    FP = []

    All = 0

    for i in range(n):
        sum = 0
        for c in range(n):
            All += confusion_matrix[i][c]
            if i != c:
                sum += confusion_matrix[i][c]
        FP.append(sum)

    FN = []

    for i in range(n):
        sum = 0
        for c in range(n):
            if i != c:
                sum += confusion_matrix[c][i]
        FN.append(sum)
    # TN = [(All - TP[i] - FP[i] - FN[i]) for i in range(n)]
    # Acc = [(TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]) for i in range(n)]
    Recall = [(TP[i]) / (TP[i] + FN[i]) if (TP[i] + FN[i]) != 0 else 0 for i in range(n)]
    Prec = [(TP[i]) / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0 for i in range(n)]
    # Spc = [(TN[i]) / (FP[i] + TN[i]) for i in range(n)]
    # Fpr = [(FP[i]) / (FP[i] + TN[i]) for i in range(n)]

    C = []
    P = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += confusion_matrix[i][j]
        C.append(sum)

    for i in range(n):
        sum = 0
        for j in range(n):
            sum += confusion_matrix[j][i]
        P.append(sum)

    prec_w = 0
    recall_w = 0
    for i in range(n):
        if (P[i] != 0):
            prec_w += (confusion_matrix[i][i] * C[i]) / P[i]
        recall_w += (confusion_matrix[i][i])

    prec_w /= All
    recall_w /= All

    macro_f = 0
    if (prec_w + recall_w != 0):
        macro_f = 2 * prec_w * recall_w / (prec_w + recall_w)

    micro_f = 0
    for i in range(n):
        if ((Prec[i] + Recall[i]) == 0):
            continue
        micro_f += (C[i] * ((2 * Prec[i] * Recall[i]) / (Prec[i] + Recall[i])))

    micro_f /= All

    print(macro_f)
    print(micro_f)


f1_score(CM)
