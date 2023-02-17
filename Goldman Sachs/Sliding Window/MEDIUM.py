from _ast import List

# 7 Length of longest substring with non - repeating characters -- Sliding Window
def lengthOfLongestSubstring(self, s: str) -> int:
    hashset = set()  # sliding window to check for duplicates
    start, maxLength = 0, 0
    for end in range(len(s)):
        while s[end] in hashset:
            hashset.remove(s[start])
            start += 1
        hashset.add(s[end])
        maxLength = max(maxLength, len(hashset))
    return maxLength



# 21 Minimum Size SubArray -- Sliding Window
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    # sliding window approach
    window, start, minLength = 0, 0, len(nums)
    for end in range(len(nums)):
        window += nums[end]
        while window >= target:
            minLength = min(end - start + 1, minLength)
            window -= nums[start]
            start += 1
    return minLength if minLength != len(nums) else 0

# 62 Maximum Size SubArray Sum - Sliding Window  -- can't be solved with sliding window..coming back to this
# after i understand prefix Sum properly
def maxSubArrayLen(self, nums: List[int], k: int) -> int:
    window_sum, start = 0, 0
    longest = 0
    for end in range(len(nums)):
        while window_sum == k:
            window_sum -= nums[start]
            start += 1
        window_sum += nums[end]
        longest = max(longest, end - start + 1)
    return longest if longest != 0 else -1
