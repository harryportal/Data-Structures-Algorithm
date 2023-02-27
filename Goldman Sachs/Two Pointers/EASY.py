from typing import List


# 1 Count Binary Strings
def countBinarySubstrings(self, s: str) -> int:
    """
    Check this for detailed explanation
    https://leetcode.com/problems/count-binary-substrings/solutions/1172569/short-easy-w-explanation-comments-keeping-consecutive-0s-and-1s-count-beats-100/
    """
    ans, prev, curr = 0, 0, 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            curr += 1
        else:
            ans += min(prev, curr)
            prev, curr = curr, 1
    ans += min(prev, curr)
    return ans

# 2 Minimum Value to Get Positive Step by Step Sum
def minStartValue(self, nums: List[int]) -> int:
    # we precompute the sum using 0 as a start value and get the minimum of the step by step sum
    # Our minimum start value should be able to make the minimum of all step by step sum equal to exactly 1
    total, minStep = 0, 0
    for num in nums:
        total += num
        minStep = min(total, minStep)
    return 1 - minStep

# 3 Reverse String
def reverseString(self, s: List[str]) -> None:
    # we simply make use of two pointers to swap the letters in place
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1