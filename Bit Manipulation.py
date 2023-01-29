import enum
import math
from typing import List


# Single Number
def singleNumber(self, nums: List[int]) -> int:
    single = nums[0]
    for i in range(1, len(nums)):
        single ^= nums[i]
    return single


# Missing Number
def missingNumber(self, nums: List[int]) -> float:
    number_sum, n = sum(nums), len(nums)
    arithmetic_sum = n * (n + 1) / 2
    return int(arithmetic_sum - number_sum)


# Sum of Two integers
""" Java Code - python has memory issues for this particular problem
class Solution {
    public int getSum(int a, int b) {
        while (b != 0) {
            int tmp = (a & b) << 1; // here we shift the bit to the left once
            a = (a ^ b);
            b = tmp;
        }
        return a;
    }
}"""


# Reverse Integer
def reverse(self, x: int) -> int:
    reverse_int = 0
    MAX = 2 ** 31 - 1
    MIN = -2 ** 31

    while x:
        # python does not handle floor division and modulo well for negatie numbers
        digit = int(math.fmod(x, 10))
        x = int(x / 10)

        # check for overflow before updating our result
        if reverse_int > MAX // 10 or (reverse_int == MAX // 10 and digit >= MAX % 10):
            return 0
        if reverse_int < MIN // 10 or (reverse_int == MIN // 10 and digit <= MIN % 10):
            return 0

        # update our reverse interger
        reverse_int = (reverse_int * 10) + digit
    return reverse_int


# Count 1 Bits
def hammingWeight(self, n: int) -> int:
    result = 0
    while n:
        n &= n - 1  # flips the least significant bit to zero
        result += 1
    return result


# Count 1 Bits
def hammingWeight(self, n: int) -> int:
    count = 0
    bitmask = 1
    for i in range(32):
        if bitmask & n:
            count += 1
        bitmask <<= 1
    return count


# Reverse Bits
def reverseBits(self, n: int) -> int:
    result = 0
    for i in range(32):
        bit = (n >> i) & 1
        result = result | bit << (31 - i)
    return result
