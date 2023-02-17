from typing import List

# Best time to buy and sell Stock
def maxProfit(self, prices: List[int]) -> int:
    max_profit, low = 0, 0
    for high in range(1, len(prices)):
        if prices[high] > prices[low]:
            max_profit = max(max_profit, prices[high] - prices[low])
        else:
            low = high
    return max_profit


# Squares of a sorted Array
def sortedSquares(self, nums: List[int]) -> List[int]:
    result = [0] * len(nums)
    left, right = 0, len(nums) - 1
    i = len(nums) - 1  # for filling the result array from the end
    while left <= right:
        if abs(nums[right]) > abs(nums[left]):
            result[i] = nums[right] ** 2
            right -= 1
        else:
            result[i] = nums[left] ** 2
            left += 1
        i -= 1
    return result


# reverse a sentence -- peace brought this up lol!
def reverse(word: str):
    result = current = ""
    space = ""
    for i in range(len(word) - 1, -1, -1):
        if word[i] != " ":
            result += space
            space = ""
            current += word[i]
        else:
            result += "".join(list(reversed(current)))
            current = ""
            space += " "
    result += "".join(list(reversed(current))) + space
    return result


# Sort Colors
def sortColors(self, nums: List[int]) -> None:
    # we sort in place
    right, left = 0, len(nums) - 1
    curr = 0
    while curr <= left:
        if nums[curr] == 0:
            # we swap with the left pointers and increase the index
            nums[right], nums[curr] = nums[curr], nums[right]
            curr += 1
            right += 1
        elif nums[curr] == 2:
            nums[left], nums[curr] = nums[curr], nums[left]
            left -= 1
        else:
            curr += 1
