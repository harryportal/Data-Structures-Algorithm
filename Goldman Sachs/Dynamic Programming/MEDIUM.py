import math
from _ast import List

# 43 Count Number of Teams -- Dynamic Programming
def numTeams(self, rating: List[int]) -> int:
    # This is a bit tricky..would have to come back to this
    n = len(rating)
    up = [0] * n
    down = [0] * n
    teams = 0
    for i in range(n):
        for j in range(i):
            if rating[j] > rating[i]:
                up[j] += 1
                teams += up[i]
            else:
                down[j] += 1
                teams += down[i]
    return teams


# 72 Maximum Product SubArray -- Dynamic Programming
def maxProduct(self, nums: List[int]) -> int:
    # maintain a max and min subarray as you iterate through the array
    currMin, curMax = 1, 1
    totalMax = max(nums)
    for num in nums:
        if num == 0:
            currMin, currMax = 1, 1
            continue
        temp = currMax
        currMax = max(currMax * num, currMin * num, num)
        currMin = min(temp * num, currMin * num, num)

        totalMax = max(totalMax, curMax)
    return totalMax



# 34 Unique Paths  -- Dynamic Programming
def uniquePaths(self, m: int, n: int) -> int:
    # build the grid
    grid = [[1] * n for _ in range(m)]
    for row in range(1, m):
        for col in range(1, n):
            grid[row][col] = grid[row - 1][col] + grid[row][col - 1]

    return grid[m - 1][n - 1]



# 35 Minimum Path Sum -- Dynamic Programming
def minPathSum(self, grid: List[List[int]]) -> int:
    # This is another dynamic programming problem that will be solved with bottom up approach
    rows, cols = len(grid), len(grid[0])
    dp = [[math.inf] * (cols + 1) for _ in range(rows + 1)]
    dp[rows - 1][cols] = 0  # come back to why i am doing rows - 1

    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            dp[row][col] = grid[row][col] + min(dp[row][col + 1], dp[row + 1][col])

    return dp[0][0]

# 28 Decode Ways  -- Dynammic Programming
def numDecodings(self, s: str) -> int:
    # Basically at every index we can either decode it as a single digit or a double digits provided that it is valid
    # The current index becomes invalid if it's equal to zero "0" and the double digits is invalid if it's more than 26
    # This will be solved recursively(2^n) but will reduced to 0(n) using memoization since the recursive functions have
    # overallaping calls
    cache = {}

    def dfs(index):
        if index == len(s):  # we found a valid decodings
            return 1
        if s[index] == "0":  # we found an invalid decoding
            return 0
        if index == len(s) - 1: return 1
        if index not in cache:
            res = dfs(index + 1)  # as a single digit
            if int(s[index:index + 2]) <= 26: res += dfs(index + 2)
            cache[index] = res
        return cache[index]

    return dfs(0)


# 27 House Robber -- Dynamic Programming-- check neetcode video on reducing the space to constant
def rob(self, nums: List[int]) -> int:
    # at every house we basically have two choices
    # 1. To rob the current house(which means we were not able to rob the previous house)
    # 2. Not rob the current house, the money we have so far is what we've up to the previous house
    # This is another dynamic programming because we are after an optimal value and the future decisions are affected
    # by earlier decisions
    n = len(nums)
    robbing = [0] * (n + 1)
    robbing[0] = nums[0]  # if you're at the first house you have a choice of robbing the house or not robbing at all
    robbing[1] = max(nums[0], nums[1])  # if you have just two houses left to rob, you rob the house that has more lol
    for i in range(2, n + 1):
        robbing[i] = max(robbing[i - 1], nums[i] + robbing[i - 2])
    return robbing[n]


# 26 Coin Change  **  -- Dynamic Programming
def coinChange(self, coins: List[int], amount: int) -> int:
    # Neetcode explains well here! https://www.youtube.com/watch?v=H9bfqozjoqs
    dp = [math.inf] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], 1 + dp[i - coin])
    result = dp[amount]
    return result if result != math.inf else -1


# 25 Delete and Earn -- dynamic programming
def deleteAndEarn(self, nums: List[int]) -> int:
    # I'll explain the logic to solving the problem using Bottom Up Approach
    maxNumber = max(nums)
    points = {i: 0 for i in range(maxNumber + 1)}
    # First we map each number to get the points we can get from it and get the maximum of the numbers at the same time
    for num in nums:
        points[num] = num + points.get(num, 0)
    maxEarnings = [0] * (maxNumber + 1)
    maxEarnings[1] = points[1]
    # The maximum you can earn at any points will the max of you taking the current price n (which means you can't take
    # the previous one (n-1) or you ignore the current the current price which means you've taken (n-1)
    for i in range(2, maxNumber):
        maxEarnings[i] = max(maxEarnings[i - 1], maxEarnings[i - 2] + points[nums[i]])
    return maxEarnings[maxNumber]


# Word Break
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    # Neetcode explains very well! https://www.youtube.com/watch?v=Sx9NNgInc3A
    dp = [False] * len(s) + 1
    dp[len(s)] = True
    for i in range(len(s)-1, -1, -1):
        for w in wordDict:
            if i + len(w) <= len(s) and s[i:i+len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    return dp[0]