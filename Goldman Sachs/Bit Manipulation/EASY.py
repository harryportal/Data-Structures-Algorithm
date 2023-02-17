from typing import List



# 17 Counting Bit
def countBits(self, n: int) -> List[int]:
    def count(num: int):
        count = 0
        while num:
            if num & 1: count += 1
            num >>= 1
        return count

    result = []
    for i in range(n + 1):
        result.append(count(i))
    return result


def countBits(self, n: int) -> List[int]:
    # A dynamic programming approach to the question to reduce the runtime to O(n)
    # https://www.youtube.com/watch?v=RyBM56RIWrM Neetcode explains here well!
    # You can talk about this if your interviewer wants you to provide a better approach
    dp = [0] * (n + 1)
    offset = 1
    for i in range(1, n + 1):
        if offset * 2 == i:
            offset = i
        dp[i] = 1 + dp[i - offset]
    return dp


# 18 Sort Integers by The Number of 1 Bits
def sortByBits(self, arr: List[int]) -> List[int]:
    # Let's count the number of 1 Bit for each number and store in a list
    def count(n):
        # we can use the flip flop method: using the AND logic on a number n and n - 1 flips the least significant
        # 1 to zero or let's use the usual method of( right shifting and AND logic )since that's more intuitive and
        # can easily be explained to the interviewer
        count = 0
        while n:  # this will continue until n is zero(runtime is O(32) -- O(1) if in worst case all the bits are 1
            if n & 1:
                count += 1
            n >>= 1
        return count

    bitsmap = []
    for i in arr:
        bitsmap.append([count(i), i])
    bitsmap.sort()  # clarify if you're allowed to use an inbuilt function since the question talks about sorting
    return [bits[1] for bits in bitsmap]