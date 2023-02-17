import math
from typing import List



# 71 3 Sum Smaller
def threeSumSmaller(self, nums: List[int], target: int) -> int:
    count = 0
    nums.sort()
    for i in range(len(nums)):
        j, k = i + 1, len(nums) - 1
        while j < k:
            valueSum = nums[i] + nums[j] + nums[k]
            if valueSum < target:
                count += k - j
                j += 1
            else:
                k -= 1
    return count


# 67  3 Sum closest -- Two pointers
def threeSumClosest(self, nums: List[int], target: int) -> int:
    """the problem would be easier to solve if you've solved the following problems
    Two Sum, Two Sum II and 3 Sum
    """
    nums.sort()
    closest = math.inf
    for i in range(len(nums) - 2):
        left, right = i + 1, len(nums) - 1
        while left < right:
            threeSum = nums[i] + nums[left] + nums[right]
            if abs(threeSum - target) < abs(target - closest):
                closest = threeSum
            elif threeSum >= target:
                right -= 1
            else:
                left += 1
    return closest

# 60 Two Sum II  -- Two Pointers
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    low, high = 0, len(numbers) - 1
    while low < high:
        value_sum = numbers[low] + numbers[high]
        if value_sum > target:
            high -= 1
        elif value_sum < target:
            low += 1
        else:
            return [low + 1, high + 1]


# 22 Container with most Water  -- Two Pointers
def maxArea(self, height: List[int]) -> int:
    # the brute force would be to have every possible combination of the heights - O(n^2)
    # we can optimize it to linear time using two pointers approach,
    # we set our left and right pointers to the first and last height respectively and we only shift the
    # pointers that have a lower height
    left, right, maximumArea = 0, len(height) - 1, 0
    while left <= right:
        area = (right - left) * min(height[left], height[right])
        maximumArea = max(maximumArea, area)
        if height[left] <= height[right]:
            left += 1
        else:
            right -= 1
    return maximumArea


# 14 Three Sum -- Two Pointers
def threeSum(self, nums: List[int]) -> List[List[int]]:
    # the brute force approach would be to use three nested for loop to check every three pairs
    # that can be formed by each interger  - O(n^3)
    # I recommend solving Leetcode's "Two Sum II" problem as the problem is just an extension of it!
    # we can reduce this runtime to 0(n^2), if we sort the array(remember the "Two Sum II" problem?)
    # pick each integer, and use the idea of (Two Sum II) question to see which two integers can add
    # up to the current integer to give us 0
    res = []
    nums.sort()  # we first sort the input array

    for i, j in enumerate(nums):
        # make sure that we're not starting with a number that has occured before cos if it has
        # it will definitely be in our result list
        if i > 0 and nums[i - 1] == j:
            continue

        # if it has not occured before, we now pick the number and find other two that will make the sum Zero
        # to do this we make use of two pointers, one will point to the index after the current index and the second
        # will point to the end of the list
        left, right = i + 1, len(nums) - 1
        while left < right:
            threesum = j + nums[left] + nums[right]
            if threesum > 0:
                # we shift the right pointer back by 1
                right -= 1
            elif threesum < 0:
                # you get the gist now lol
                left += 1
            else:  # we found a three numbers that sun up to zero
                res.append([j, nums[left], nums[right]])
                # we move our left pointer forward at this point and make sure
                # we don't move it to a position containing a value that has already been used
                left += 1
                while nums[left] == nums[left - 1] and left < right:
                    left += 1
        return res


# 9 Longest Palindromic Substring -- Two Pointers
def longestPalindrome(self, s: str) -> str:
    # the idea is to take each character in the string and expand around the centers
    # we would need to consider cases for odd even length
    result = ""
    maximum_length = 0

    def check(left, right):
        while left >= 0 and right < len(s) and s[right] == s[left]:
            length = right - left + 1
            if length > maximum_length:
                result = s[left:right + 1]
                maximum_length = result
            left -= 1
            right += 1

    for i in range(len(s)):
        # odd palindrome
        check(i, i)

        # even palindrome
        check(i, i + 1)
    return result


