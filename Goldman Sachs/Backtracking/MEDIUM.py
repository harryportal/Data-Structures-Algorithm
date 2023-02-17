from typing import List

# 85 Subsets -- Backtracking
def subsets(self, nums: List[int]) -> List[List[int]]:
    result = []

    def backtrack(index, subset):
        if index >= len(nums):
            result.append(subset)  # base case
            return

        # decision Tree
        backtrack(index + 1, subset + nums[index])
        backtrack(index + 1, subset)

    backtrack(0, [])
    return result


# 86 Combination Sum --  Backtracking
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    result = []

    def backtrack(index, path, remaining):
        if remaining == 0:
            result.append(path)
            return

        elif remaining < 0:
            return

        for i in range(index, len(candidates)):
            backtrack(i, path + [candidates[i]], target - candidates[i])

    backtrack(0, [], target)


# 64 Permutations -- Bactkrackig
def permute(self, nums: List[int]) -> List[List[int]]:
    result = []
    if len(nums) == 1:  # base case
        return [nums[:]]

    for _ in range(len(nums)):
        temp = nums.pop(0)
        permutations = self.permute(nums)

        for perm in permutations:
            perm.append(temp)
        result.extend(permutations)
        nums.append(temp)
    return result

# 33 Generate Paranthesis  -- Backtracking
def generateParenthesis(self, n: int) -> List[str]:
    # This is a backtracking problem
    result = []

    def backtrack(opened, close, current):
        if opened == close == n:
            result.append(current)
            return

        if opened < n:
            backtrack(opened + 1, close, current + "(")

        if close < opened:
            backtrack(opened, close + 1, current + ")")

    backtrack(0, 0, "")
    return

# 23 Word Search   --  Backtracking
def exist(self, board: List[List[str]], word: str) -> bool:
    # This can be solved with backtracking
    rows, cols = len(board), len(board[0])

    # visited = set()  once we use a character once during our backtracking, we can't use it again

    def dfs(row, col, index):
        if index == len(word):
            return True
        if col not in range(cols) or row not in range(rows) or board[row][col] != word[index] or \
                board[row][col] == ".":
            return False

        temp, board[row][col] = board[row][col], "."
        result = dfs(row + 1, col, index + 1) or \
                 dfs(row - 1, col, index + 1) or \
                 dfs(row, col + 1, index + 1) \
                 or dfs(row, col - 1, index + 1)
        board[row][col] = temp
        return result
    for row in range(rows):
        for col in range(cols):
            if dfs(row, col, 0): return True
    return False



# 78 Letter Combinations of a Phone Number -- Backtracking
def letterCombinations(self, digits: str) -> List[str]:
    result = []
    letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
               "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    if not digits:
        return []

    def backtrack(index, currStr):
        if len(currStr) == len(digits):
            result.append(currStr)
            return

        for char in letters[digits[index]]:
            backtrack(index + 1, currStr + char)

    backtrack(0, "")
    return result