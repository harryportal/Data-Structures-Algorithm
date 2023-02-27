from _ast import List

# 68 Find Peak Element -- Binary Search
def findPeakElement(self, nums: List[int]) -> int:
    """Solving this with binary search is quite tricky
    This is a quick breakdown..Note the interviewer will definitely not hint you about binary search so it's okay
    to talk about the linear approach which is too easy for an interview question.
    Basically when doing our binary search here, we're just trying to cut a section of the array that does'nt need to
    be checked.
    if we start from the middle of the list, we check if the number at the middle is more than the number directly after
    it. If that is the case, the number at the middle is a potential peak element so we don't need to check the array
    starting from the number after the current middle number. Reverse is the case if the number at the middle is
    less than the number just after it."""
    low, high = 0, len(nums) - 1
    while low < high:
        mid = (low + high) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return low


# 59 Search in rotated Sorted Array -- Binary Search
def search(self, nums: List[int], target: int) -> int:
    """ There are a few things to note about rotated sorted arrays
    1. We still apply our binary search algorithm since they are still kinda sorted
    2. The array will have two sorted portion(left and right) and knowing the portion we are guides how we shift our low
    and high pointers
    3. Every value in the right portion are less than the values in the kleft portion
    """
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] >= nums[low]:  # we're in the left sorted portion
            if target < nums[low] or target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        else:  # we're in the right sorted portion
            if target < nums[mid] or target > nums[high]:
                high = mid - 1
            else:
                low = mid + 1
    return -1  # Target was not found


# 80 Find K Closest Elements --  Binary Search
def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    # we simply use binary search to search for the left most index closer to x and return the 'k' lenght subarray
    # starting from that left index
    low, high = 0, len(arr) - k
    while low < high:
        mid = (low + high) // 2
        if x - arr[mid] > arr[mid + k] - x:
            # we shift our left pointer forward
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]

# H Index II
def hIndex(self, citations: List[int]) -> int:
    low, high = 0, len(citations) - 1
    n = len(citations)
    while low <= high:
        mid = (low + high) // 2
        if citations[mid] == n - mid:
            return  n - mid
        elif citations[mid] < n - mid:
            low = mid + 1
        else:
            high = mid - 1
    return n - low



# 20 Find Minimum in a rotated Sorted Array  -- Binary Search
def findMin(self, nums: List[int]) -> int:
    # we use a modified binary search algorithm since the array is still somehow rotated
    # https://www.youtube.com/watch?v=nIVW4P8b1VA ,Neetcode explains well here!
    left, right = 0, len(nums) - 1
    if len(nums) == 1: return nums[0]
    minimum = nums[0]
    while left <= right:
        if nums[left] <= nums[right]:  # the array is properly sorted at this point
            return min(minimum, nums[left])
        mid = (left + right) // 2
        minimum = min(minimum, nums[mid])
        if nums[mid] >= nums[left]:  # we search to the right when we're in the left sorted portion
            left = mid + 1
        else:  # we search to the left when we're in the left sorted portion of the array
            right = mid - 1


# 15 Search a 2D Matrix -- Binary Search
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    # the brute force(O(n^2)) approach is the most intuitive approach
    # However we can take advantage of the sorted property of the matrix and use
    # binary search to search through the matrix
    # first we figure out the row to search
    rows, cols = len(matrix), len(matrix[0])
    low, high = 0, rows - 1
    while low <= high:
        row = (low + high) // 2
        if target > matrix[row][-1]:  # we search the rows down if the target is more than the value in the current row
            low = row + 1
        elif target < matrix[row][0]:  # we search to the top if the target is less than the first number in the row
            high = row - 1
        else:
            break

    # now we've found the row
    row = (low + high) // 2
    low, high = 0, cols - 1
    while low <= high:
        mid = (low + high) // 2
        if target > matrix[row][mid]:
            low = mid + 1
        elif target < matrix[row][mid]:
            high = mid - 1
        else:
            return True
    return False

