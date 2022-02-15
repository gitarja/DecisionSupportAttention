def solution(A, K):
    n = len(A)
    best = 0
    count = 1
    for i in range(n - K - 1):
        if (A[i] == A[i + 1]):
            count = count + 1
        else:
            count = 1
        best = max(best, count)
    result = max(best + K, min(K+1, n))

    return min(result, n)


A = [1, 2, 3, 3, 3, 3, 3]
K = 8

print(solution(A, K))
