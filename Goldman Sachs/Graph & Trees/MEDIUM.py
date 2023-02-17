from _ast import List
from typing import Optional
from Learning.Trees import TreeNode


# 87 All Nodes distance K in binary Tree
def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
    # we convert the tree to a graph(intution being that we can travel reverse if we were to stick with it being a tree)
    # and the node at distance k from the current node can be before after it.
    # After converting it to a graph, we do a simple bfs
    graphDict = {}

    def graph(node, parent):
        if not node:
            return

        if not graphDict[node]:
            graphDict[node] = []

        if parent:
            graphDict[node].append(parent)

        if node.left:
            graphDict[node].append(node.left)
            graph(node.left, node)

        if node.right:
            graphDict[node].append(node.right)
            graph(node.right, node)

    # bfs
    queue, visited = [[target, 0]], set()
    result = []
    while queue:
        node, depth = queue.pop(0)
        visited.add(node)

        if depth == k:
            result.append(node.val)

        for neighbour in graphDict[node]:
            if neighbour not in visited:
                queue.append([neighbour, depth + 1])

    return result


# 88 Balance a Binary Search Tree -- Trees, Inorder, Binary Search
def balanceBST(self, root: TreeNode) -> TreeNode:
    """ Perform an inorder traversal to store the nodes in a list
     Build the binary search tree recursively by using binary search algorithm """

    def inorder(node):
        # let's do a recursive inorder since it has few lines of code compared to iterative
        if not node:  # base case
            return []
        left, right = inorder(node.left), inorder(node.right)
        return left + [node.val] + node.right

    nodes = inorder(root)

    def binarySearch(left, right):
        # build tree recursively
        if left > right:
            return None
        mid = (left + right) // 2
        root = TreeNode(nodes[mid])
        root.left = binarySearch(left, mid - 1)
        root.right = binarySearch(mid + 1, right)
        return root

    return binarySearch(0, len(nodes) - 1)


# 89 Count Complete Tree Nodes
def countNodes(self, root: Optional[TreeNode]) -> int:
    """ Quite tricky, I'll provide a detailed explanation soon
    ps: I didn't figure this out myself, had to read the official solution
    Just note that the number of nodes in a complete tree equals 2^d - 1 + the nodes on the last level which are
    filled as far left as possible """
    # compute the depth of the tree
    node, depth = root, 0
    if not node: return 0

    while node.left:
        node = node.left
        depth += 1

    if depth == 0: return 1

    # binary search function to check if a node value exists
    def check(index, node):
        left, right = 0, pow(2, depth) - 1
        for i in range(depth):
            mid = (left + right) // 2
            if index <= mid:
                node = node.left
                right = mid
            else:
                node = node.right
                left = mid + 1
        return node is not None

    left, right = 1, pow(2, depth) - 1
    while left <= right:
        mid = (left + right) // 2
        if check(mid, root):
            left = mid + 1
        else:
            right = mid - 1

    return pow(2, depth) - 1 + left



# 69 Insert into a Binary Tree
def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """For a binary search tree, every node is more than the all the values in it left subtree and less than all the
    values in it right subtree"""
    node = root
    while node:
        # if the value is more than the current node value, we either insert it as the right child if it has none or
        # we continue searching from the right child
        if val > node.val:  # we check the right subtree
            if not node.right:
                node.right = TreeNode(val)
                return root
            else:
                node = node.right
        else:
            if not node.left:
                node.left = TreeNode(val)
                return root
            else:
                node = node.left


# 70 Binary Search Iterator
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        current = root
        while current:
            self.stack.append(current)
            current = current.left

    def next(self) -> int:
        result = self.stack.pop()
        current = result.right
        while current:
            self.stack.append(current)
            current = current.left
        return result.val

    def hasNext(self) -> bool:
        return len(self.stack)


# 66 Snake and Ladders -- Bfs(shortest path)
def snakesAndLadders(self, board: List[List[int]]) -> int:
    lenght = len(board)
    board.reverse()  # reverse the board makes our calculation easier

    def getPosition(value):  # an helper function to get the row and column if the value is not -1
        row = (value - 1) // lenght
        col = (value - 1) % lenght
        if row % 2:  # extra logic since the rows are alternating
            col = lenght - col - 1
        return [row, col]

    queue = [[1, 0]]  # [value,moves]
    visited = set()  # so we don't visit a position twice
    while queue:
        value, moves = queue.pop(0)
        for i in range(1, 7):  # let's throw a die
            nextValue = value + i
            row, col = getPosition(nextValue)
            if board[row][col] != -1:  # then there is a snake or ladder
                nextValue = board[row][col]
            if nextValue == lenght ** 2:  # we've gotten to the end of the board
                return moves + 1
            if nextValue not in visited:
                queue.append([nextValue, moves + 1])
                visited.add(nextValue)
    return -1



# 46 Binary Tree Right Side View -- Trees
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    # we basically do a bfs and return the node at the right most side of each level
    if not root: return []  # edge case
    queue, result = [root], []
    while queue:
        for i in range(len(queue)):  # go through each level
            node = queue.pop(0)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(node.val)
    return result

# 24 Rotting Oranges  -- Graph(Bfs)
def orangesRotting(self, grid: List[List[int]]) -> int:
    # This problems becomes easier when you view it as a graph and imagine the rotting oranges are somehow the
    # roots of the graph
    # we'll use a bfs approach here(not dfs) since we can have multiple roots(rotting oranges)
    rows, cols = len(grid), len(grid[0])  # just like every grid problem
    fresh_oranges, time = 0, 0
    # get the number of fresh oranges and append the dimensions of the rotten oranges to the queue that will be used
    # for our bfs algorithm
    queue = []
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 2:
                queue.append([row, col])
            if grid[row][col] == 1:
                fresh_oranges += 1

    # time for our bfs logic
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # for checking the adjacent oranges
    while queue and fresh_oranges > 0:
        for i in range(len(queue)):
            row, col = queue.pop(0)
            for r, c in directions:
                if row + r not in range(rows) or col + c not in range(cols) or grid[row + r][col + c] != 1:
                    continue
                grid[row + r][col + c] = 2
                queue.append([row + r, col + c])  # add the dimension since the orange is also now rotten
                fresh_oranges -= 1
        time += 1
    return time if fresh_oranges == 0 else -1



# 13 Number of Islands
def numIslands(self, grid: List[List[str]]) -> int:
    # we use a depth first search to search recursively the neighbours(vertically and horizontally)
    # whenever we get to a position of the island(grid) that has a value of 1
    # we can also use a bfs here but i think the dfs code is more concise
    rows, cols = len(grid), len(grid[0])
    num_islands = 0

    def dfs(row, col):
        if not (row in range(rows) and col in range(cols)) or grid[row][col] != "1": # edge case
            return
        # mark the position as visited by changing the value (if the interveiwer permits the input grid modified)
        # or use a hashset to keep track of visited positions
        grid[row][col] = "*"
        dfs(row + 1, col)
        dfs(row - 1, col)
        dfs(row, col + 1)
        dfs(row, col - 1)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "1":
                dfs(row, col)
                num_islands += 1

    return num_islands