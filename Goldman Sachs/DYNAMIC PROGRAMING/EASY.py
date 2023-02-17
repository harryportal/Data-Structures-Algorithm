# 14 Fibonnaci Sequence
def fib(n: int) -> int:
    # The most common and inefficient approach is to do this recursively -- O(2^n)
    # if n <= 1:
    #     return n
    # return fib(n - 1) + fib(n - 2)

    # dynamic programming approach -- bottom up iteration
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

    # dynamic programming approach  II - Top down memoization
    # cache = {}
    # if n <= 1: return n
    # if n not in cache:
    #   cache[n] = fib(n-1) + fib(n-2)
    # return cache[n]

# 19 Climbing Stairs
def climbStairs(self, n: int) -> int:
    # If you read the problem you'll notice a recurrence relation that the number of ways to get to let's say the
    # the third step is the same as the sum of the number of ways to get to the 2nd step and the first step
    # dp[i] = dp[i-1] + dp[1-2].. Therefore we can solve this problem with either bottom up or top down dynamic pro
    # gramming approach

    # Bottom Up Approach
    if n <= 2: return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

    # Top Down Approach with memoization
    hashmap = {}

    def dp(n):
        if n <= 2: return n
        if n not in hashmap:
            hashmap[n] = dp(n - 1) + dp(n - 2)
        return hashmap[n]

    return dp(n)


