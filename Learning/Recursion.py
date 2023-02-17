from typing import List
from Linkedlist import ListNode


# Reverse String
def reverseString(self, s: List[str]) -> None:
    # Solve this with recursion and constant space complexity
    # This is a constant time space because there is not call stack ( the operation is performed
    # before the recursion is done)
    # define an helper function
    def reverse(string, start, end):
        if start >= end:
            return
        # swap the positions
        string[start], string[end] = string[end], string[start]
        # call the function recursively
        reverse(string, start + 1, end - 1)

    reverse(s, 0, len(s) - 1)


# Swap node in Pairs
def swapPairs(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head

    # Swap the nodes
    first_node = head
    second_node = head.next

    # call the function recursively on the next pair of nodes to be swapped
    first_node.next = swapPairs(second_node.next)
    second_node.next = first_node

    # Space here is 0(n) but can be optimisized to 0(1) using dummy pointers
    return second_node


def findSubsequences(self, nums: List[int]) -> List[List[int]]:
    # This will be solved with backtracking
    result = set()  # to make sure each subsequence is unique

    def backtrack(index, sequence):
        if len(sequence) >= 2:
            result.add(tuple(sequence))
        if index == len(nums):  # base case
            return
        if not sequence or sequence[-1] <= nums[index]:
            backtrack(index + 1, sequence + [nums[index]])
        backtrack(index + 1, sequence)  # call the recursive to start from the next index

    backtrack(0, [])
    return list(result)


def restoreIpAddresses(self, s: str) -> List[str]:
    # This will be solved with backtracking again because we have to make sort a permutation of choices
    # that satisifes the question's criteria
    result = []

    if len(result) > 12:  # edge case since leetcode says we can have a list of up to 1000 characters
        return []

    # an helper function
    def bactrack(index, dots, currIP):
        if dots == 4 and index == len(s):
            result.append(currIP[:-1])  # base case plus we don't add the last dot
        if dots > 4:  # we can't continue along that decision branch
            return
        for j in range(index, min(index + 3, len(s))):
            # the number must be less than 256, it must either be of lenght 1 or the first digit must not be zero
            if int(s[index:j + 1]) <= 255 and (index == j or s[index] != "0"):
                bactrack(j + 1, dots + 1, currIP + s[index:j + 1])

    bactrack(0, 0, "")
    return result


# Palindromic Subsequence
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result = []

        def dfs(index, path):
            if index >= len(s):  # base case
                result.append(path)
            for j in range(index, len(s)):
                if self.isPalindrome(s, index, j):
                    dfs(j + 1, path + s[index:j + 1])

        dfs(0, [])
        return result

    def isPalindrome(self, s, left, right) -> bool:
        # you already know what this is supposed to do
        while left <= right:
            if s[left] != s[right]:
                return False
            right -= 1
            left += 1
        return True
