# 3 Pascals Triangle
from typing import List


def generate(self, numRows: int) -> List[List[int]]:
    # runtime - O(n^2), space - in the worst case our temp variable would hold n numbers, O(n)
    result = [[1]]
    for i in range(numRows - 1):
        temp = [0] + result[-1] + [0]
        row = []
        for j in range(len(temp) - 1):
            row.append(temp[j] + temp[j + 1])
        result.append(row)
    return result

# 8 Merge Sorted Arrays
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    last_index = m + n - 1
    m, n = m - 1, n - 1
    while m >= 0 and n >= 0:
        if nums1[m] > nums2[n]:
            nums1[last_index] = nums1[m]
            m -= 1
        else:
            nums1[last_index] = nums2[n]
            n -= 1
        last_index -= 1
    # if there are still values in nums2
    while n >= 0:
        nums1[last_index] = nums2[n]
        n -= 1
        last_index -= 1


# 12 Find Pivot Index
def pivotIndex(self, nums: List[int]) -> int:
    # the naive approach would be that as we go through the array,
    # For every index, we compute sum of the values to the right and also do the same for the values to the left
    # if the two sum is equal, we return the current index
    # However this is 0(n^2) since we are computing a sum(O(n)) in every iteration
    # to reduce the runtime to O(n), we can compute the left sum as we move through the array
    # to get the right sum, we subtract the left sum and current value from the total sum of the array
    leftSum, total = 0, sum(nums)
    for i in range(len(nums)):
        rightSum = total - leftSum - nums[i]
        if rightSum == leftSum:
            return i
        leftSum += nums[i]
    return -1

# 44 Perfom String Shifts **
def stringShift(self, s: str, shift: List[List[int]]) -> str:
    # the brute force would be to modify the string based on each number of shifts and direction
    # we can optimize this further by computing the net shifts and applying this shifts only once
    leftShifts = 0
    for direction, amount in shift:
        if direction == 0:
            leftShifts += amount
        else:
            leftShifts -= amount

    # there are two cases here now, it's either the leftshift amount is negative which means that the right shift
    # was more or the number of shifts is more than the length of the string
    # To take of these two scenarios, we just mod the result by length of the string
    leftShifts %= len(s)
    stringShift = s[leftShifts:] + s[:leftShifts]
    return stringShift



# 46 Count Negative in a sorted Matrix
def countNegatives(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    r, c, count = rows - 1, 0, 0

    # we go buttom up  -- last row, first column
    # if the current number is negative, everything else to the left of the the row will be negative
    # Hence we move to the upper row
    while r >= 0 and c < cols:
        if grid[r][c] < 0:
            count += cols - c
            r -= 1
        else:
            c += 1
    return count



# 37 Excel Sheet Column Number
def titleToNumber(self, columnTitle: str) -> int:
    result = 0
    for i in range(len(columnTitle)):
        result *= 26
        # we can use ASCII conversion to get the character number
        result += ord(columnTitle[i]) - ord('A') + 1
    return result


# 31 Move Zeroes
def moveZeroes(self, nums: List[int]) -> None:
    # One approach is to create two arrays, store the zeroes in ones and the others in the second one and join the two
    # lists. We can optimize the space complexity to 0(1) by using the knowledge of quick sort algorithm to partition
    # also seeing the question as "moving the non zeroes to the front" rather than "moving the zeros to the back" makes
    # the solution more intuitive
    left = 0
    for right in range(len(nums)):
        if nums[right] > 0:
            nums[right], nums[left] = nums[left], nums[right]
            left += 1


# 21 Greatest Common Divisor of Two Strings
def gcdOfStrings(self, str1: str, str2: str) -> str:
    # we can approach this mathematically
    # if two numbers have a common divisor, this means that two numbers can be expressed as a multiple of that divisor
    # so looking at the below example
    # a = 4, b = 6..we see that ab (4 * 6)->[2*2*2*2*2] = ba (6 * 4)->[2*2*2*2*2]
    # Therefore two strings would have a common divisor if the concantentation of the two strings are equal in both ways
    # stringa + stringb = stringb + stringa
    # Once we know this, we just need to find the gcd of the two strings lenght and return the string up to that length
    if str1 + str2 != str2 + str1:
        return ""
    gcdLength = math.gcd(len(str1), len(str2))
    return str1[:gcdLength]


def getRow(self, rowIndex: int) -> List[int]:
    # so the first approach we can use here is to simple generate r number of rows for the pascal triangle
    # and then return the last row in our list as the answer -- rutime O(n^2)
    # but that can be optimized using some basic knowledge of maths and combinations
    # Here are few things to note
    # The kth row in a pascal triangle will have k + 1 numbers
    # to get a value at a row or column in a pascal triangle can be done using combination
    # value at nth row and kth column = nCk = n!/(n-k)!k! == n*(n-1)*...(n-k+1)/k!
    # e.g 5C4 == 5*4*3*2/1*2*3*4
    # so basically we can initiliase the numerator to the row number given and denominator to 1
    # and then basically decrease the numerator and increase the numerator as we try to figure the value at each column
    # of the rth row
    # use the link below for a better explanation maybe
    # https://leetcode.com/problems/pascals-triangle-ii/solutions/1203260/very-easy-o-n-time-0-ms-beats-100-simple-maths-all-languages/
    row = [1] * (rowIndex + 1)
    up, down = rowIndex, 1
    for i in range(1, rowIndex):
        row[i] = int(row[i - 1] * up / down)
        up -= 1
        down += 1
    return row


# 7 Rotate Strings
def rotateString(self, s: str, goal: str) -> bool:
    # the first thing you migth think of is to split the first string into array
    # and keep keep rotating within the range of its length, then for every rotation
    # we compare it with second string, we return False if there was no match

    # another approach is to double the first string, by doubling it gives every posible string
    # that would have been gotten by rotating it one by one
    # now we only need to check if the new string contains the second string
    return len(s) == len(goal) and (goal in s + s)

    # using the in operator to check here has a runtime of O(n^2), to reduce this to o(n) you can
    # a popular string matching algorithm called KMP(Kunnath...whatever)
    # the complexity is an overkill for a leetcode "Easy" question


# 40 Power of Three
def isPowerOfThree(self, n: int) -> bool:
    if n <= 0: return False
    while n % 3 == 0:
        n //= 3
    return n == 1