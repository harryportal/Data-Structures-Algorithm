from typing import List

# 84 Minimum moves to Burst Balloons -- Greedy Algorithm
def findMinArrowShots(self, points: List[List[int]]) -> int:
    points.sort(key=lambda x: x[1])  # sort the points by the end values
    arrows = 1  # you need at least one arrow
    currentEnd = points[0][1]  # start with the first balloon end value
    for i in range(1, len(points)):
        start, end = points[i]
        # if the current balloon's start value is more than the previous ballon
        # end value, we need to increase the number of arrows since they do not overlap
        if start > currentEnd:
            arrows += 1
            currentEnd = end
    return arrows


# 58 Gas Stations -- Greedy Algorithm
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    total_tank, curr_tank = 0, 0
    starting_index = 0
    for i in range(len(gas)):
        remaining = gas[i] - cost[i]
        total_tank += remaining
        curr_tank += remaining
        if curr_tank < 0:
            starting_index = i + 1
            curr_tank = 0
    return starting_index if total_tank >= 0 else -1



# 42 Max Diff you can get from changing an Integer  -- Greedy Algorithm
def maxDiff(self, num: int) -> int:
    # We can approach this with greedy algorithm the max integer from changing the integer would be changing the
    # first digit in the number that is not's a 9 to a 9 the min integer would be changing either the first integer(
    # if it's not already a 1) or changing the next digits to zero if it's not already a zero.. we can't change the
    # first digit to zero since we don't want to have leading zeroes
    num = str(num)
    maxInt = num
    for x in num:
        if x != '9':
            maxInt = num.replace(x, '9')
            break

    minInt = num
    for x in num:
        if num[0] == x and x != '1':  # change the first integer to 1 if it's not already 1
            minInt = num.replace(x, '1')
            break
        if num[0] != x and x != '0':  # else change the next zero digit to a zero
            minInt = num.replace(x, '0')
            break
    return int(maxInt) - int(minInt)


# 40 Jump game -- Greedy Algorithm
def canJump(self, nums: List[int]) -> bool:
    goal = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= goal:
            goal = i

    return goal == 0


# 19 Maximum SubArray -- Greedy
def maxSubArray(self, nums: List[int]) -> int:
    currSum = 0
    maxSum = 0
    for num in nums:
        if currSum < 0:  # if the sum of the previous subarray is negative, we reset it to zero
            currSum = 0  # cos we are really after getting the maximum subarray
        currSum += num
        maxSum = max(maxSum, currSum)
    return maxSum


# 17 Meeting Rooms   -- Greedy
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    # basically for every meeting, we need to know if it starts a time less than the end time of another
    # meeting, if this happens we know the two meetings would overlap so we allocate a seperate room for the
    # current meeting
    # since it's obvious we want to make the best use of the number of rooms we allocate, we have to be
    # greedy with this approach
    # let's seperate the meetings into start time and end time and sort them seperately
    starting_times = sorted([time[0] for time in intervals])
    ending_times = sorted([time[1] for time in intervals])

    rooms, start, end = 0, 0, 0
    maxrooms = 0
    while start < len(intervals):
        if starting_times[start] < ending_times[end]:
            # the starting time of the current meeting overlaps with the ending time of another meeting
            rooms += 1
            start += 1
        else:
            rooms -= 1
            end += 1
        maxrooms = max(rooms, maxrooms)
    return maxrooms


# 4 Palindromic Substring -- Two Pointers
def countSubstrings(self, s: str) -> int:
    # The most intuitive approach is to take each character and check it against the remaining characters to
    # see if it forms a palindrome. However the runtime is O(n^3)[O(n^2) for nested for loop * O(n) for checking
    # if it forms a palindrome
    # we can optimize the runtime to O(n^2) by assuming each character in the string is a middle of another subtring
    # and then expands towards the left and right( we do this for odd and even lenght palindrome)
    result = 0

    def countpalindrome(left, right):  # an helper
        nonlocal result
        while left >= 0 and right < len(s) and s[left] == s[right]:
            result += 1
            left -= 1
            right += 1

    for index in range(len(s)):
        countpalindrome(index, index)  # odd palindromic length
        countpalindrome(index, index + 1)  # even palindromic length

    return result

# 6 Break a Palindrome -- Greedy
def breakPalindrome(self, palindrome: str) -> str:
    # we want to be greedy with our approach here
    # basically to break a palindrome in the least lexological order, we want to change the first letter we see to
    # 'a'(that's if it not 'a' already).. the edge case is if it happens that all the letter are a then we want to
    # change the last letter to b
    length = len(palindrome)
    if length == 1:
        return palindrome
    for i in range(length // 2):  # going through half of the string since it's a palindrome
        if palindrome[i] != "a":
            return palindrome[:i] + "a" + palindrome[i + 1:]
    return palindrome[:-1] + "b"


# 2 Best time to buy and sell stock II -- Greedy
def maxProfit(self, prices: List[int]) -> int:
    # we only buy when we know the price would rise the next day
    maxProfit = 0
    for index in range(len(prices) - 1):
        if prices[index] < prices[index + 1]:
            maxProfit += prices[index + 1] - prices[index]
    return maxProfit


