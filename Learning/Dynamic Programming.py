from typing import List


# House Robber
def rob(self, nums: List[int]) -> int:
    cache = {}

    def dp(n):
        if n == 0:
            return nums[0]
        if n == 1:
            return max(nums[1], nums[0])
        if n not in cache:
            cache[n] = max(dp(n - 1), (dp(n - 2) + nums[n]))
        return cache[n]

    return dp(len(nums) - 1)






# Min Cost Climbing Stairs
def minCostClimbingStairs(self, cost: List[int]) -> int:
    n = len(cost)
    minimum = [0] * (n + 1)
    for i in range(2, n + 1):
        one_step = minimum[i - 1] + cost[i - 1]
        two_steps = minimum[i - 2] + cost[i - 2]
        minimum[i] = min(one_step, two_steps)
    return minimum[-1]


# Tribonacci Number
def tribonacci(self, n: int) -> int:
    if n == 1 or n == 2:
        return 1
    if n == 0:
        return 0
    cache = [0] * (n+1)
    cache[1] = cache[2] = 1
    for i in range(3, n+1):
        cache[i] = cache[i-1]+cache[i-2]+cache[i-3]
    return cache[n]