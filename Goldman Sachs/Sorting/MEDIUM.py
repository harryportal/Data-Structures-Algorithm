from _ast import List
from functools import cmp_to_key

# 16 Largest Numbers ** check leetcode implementation on custom sorting without using the functools module -- sorting
def largestNumber(self, nums: List[int]) -> str:
    # for each pairwise comparison during the sort, we compare the
    # numbers achieved by concatenating the pair in both orders
    nums = [str(i) for i in nums]  # convert all the integers to strings

    def compare(n1, n2):  # define a custom sorting function
        if n1 + n2 > n2 + n1:
            return -1
        else:
            return 1

    nums.sort(key=cmp_to_key(compare))  # clarify with your interviewer if you're allowed to use an imported module

    # take care of a minor edge case where all the numbers are 0
    # it should return just zero if that's the case
    return "".join(nums) if nums[0] != "0" else "0"

# 38 Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts  -- sorting
def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
    # If you observe carefully, you would realize that the maximum area of area of cake has a width that is equal
    # to the maximum width gotten after applying only the vertical cuts and has a height equal to the maximum
    # height after applying only the horizontal cuts
    # we first sort the array to ensure the cuts are beside each other
    horizontalCuts.sort()
    verticalCuts.sort()

    # get the maximum height
    # consider the edges first
    max_height = max(horizontalCuts[0], h - horizontalCuts[-1])
    for i in range(1, len(horizontalCuts)):
        max_height = max(horizontalCuts[i] - horizontalCuts[i - 1], max_height)

    # get the maximum width
    # consider the edges first
    max_width = max(verticalCuts[0], w - verticalCuts[-1])
    for i in range(1, len(verticalCuts)):
        max_width = max(verticalCuts[i] - verticalCuts[i - 1], max_width)

    return (max_height * max_width) % (10 ** 9 + 7)


# 79 Sort the Jumbled Numbers -- Arrays, Strings
def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
    result = []
    for value in nums:
        string = str(value)
        newString = ""
        for char in string:
            newString += str(mapping[int(char)])
        result.append([int(string), int(newString)])
    result.sort(key=lambda x: x[1])
    return [i[0] for i in result]


# 73 Minimum moves to make array elements equal  -- Sorting
def minMoves(self, nums: List[int]) -> int:
    nums.sort()
    moves = 0
    for i in range(len(nums) - 1, -1, -1):
        moves += nums[i] - nums[0]
    return moves