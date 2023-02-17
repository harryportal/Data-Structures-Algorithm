from typing import Optional, List

"""
Full Tree - every node except for the leaf node has 0 or 2 trees
Complete Tree -  every level is completely filled except for the last node which  can be filled from 
left to right  ( to remember use the idea of completely filled to remember then you can easily remember the reamining
two lol )!
Perfect Tree: This is a tree where all nodes except for the leaf node has 2 children

"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Learning how BST Tree works
class BinarySearchTree:
    def __init__(self, root_node: TreeNode):
        self.root_node = root_node

    def findMininum(self):  # -- O(h) where h is the height of the tree
        # keep searching the left subtree iteratively
        current = self.root_node
        while current.left:
            current = current.left
        return current

    def findMinimumRecursive(self, node: TreeNode):
        # keep searching the left subtree recursively
        if not node.left:  # base case
            return node
        self.findMinimumRecursive(node.left)

    def insert(self, node: TreeNode):
        # we go through the left or right subtree
        # we keep track of the current node and it's parent
        current, parent = self.root_node, None
        while current:
            parent = current
            if node.val <= current.val:
                current = current.left
            else:
                current = current.right
        # at this point, the parent would have no children
        if node.val <= parent:
            parent.left = node.val
        else:
            parent.right = node.val

    # omo that delete method choke, i no sabi am
    def search(self, node: TreeNode):
        current = self.root_node
        while current:
            if current.val == node.val:
                return True
            elif current.val < node.val:
                current = current.left
            else:
                current = current.right
        return False  # if the code gets here, then the node is not in the BST

    # Tree traversal methods
    # 1. Depth First search
    def inorder(self, node):
        visited = []
        current = node
        if not current: return
        self.inorder(current.left)
        visited.append(current.val)


# Easy
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    # we visit the current node, its left subtree then right subtree
    # let's do it recursively
    answer = []

    def dfs(node: TreeNode):
        if not node: return  # base case
        answer.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return answer


# Same question but now iteratively using a stack
def preorderTraversal2(self, root: Optional[TreeNode]) -> List[int]:
    if not root: return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)
    return result


def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    # recursively
    answer = []

    def dfs(node):
        if not node: return
        dfs(node.left)
        answer.append(node.val)
        dfs(node.right)

    dfs(root)
    return answer


def inorderTraversal2(self, root: Optional[TreeNode]) -> List[int]:
    # iteratively
    stack, result, curr = [], [], root
    while stack or curr:
        while root:
            nonlocal curr
            stack.append(root)
            curr = root.left
        if not stack: return result
        node = stack.pop()
        result.append(node.val)
        curr = node.right


# Invert Binary Tree
def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    # this will be done recursively
    # swap the nodes and call the function on the left and right node
    if not root:  # base case
        return None
    root.left, root.right = root.right, root.left
    invertTree(root.left)
    invertTree(root.right)
    return root


# Maximum depth of a binary Tree
def maxDepth(self, root: Optional[TreeNode]) -> int:
    # we can use a bfs to count and return the number of levels in the tree
    if not root:
        return 0
    queue = [root]
    count = 0
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)
        count += 1
    return count


def maxDepth2(root: Optional[TreeNode]) -> int:
    """
    we do with a one line of code using recursive dfs
    you actually want to avoid doing this in an interview
    except you can explain the time and space complexity
    """
    if not root: return 0
    return max(maxDepth2(root.left), maxDepth2(root.right)) + 1  # adding + 1 to account for the root


# Diameter of a Binary Tree
def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    res = [0]

    def dfs(root):
        if not root:
            return -1
        left = dfs(root.left)  # computes the height of left subtree
        right = dfs(root.right)  # computes the height of right subtree

        res[0] = max(res[0], 2 + left + right)

        return 1 + max(left, right)

    dfs(root)
    return res[0]


#  Binary Tree Level Order Traversal
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    # we use a breadth first search here
    if not root: return []  # edge case
    queue = [root]
    result = []
    while queue:
        for i in range(len(queue)):
            current_level = []
            node = queue.pop(0)
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
    return result


# Lowest Common Ancestor
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # we look for the node at which p and q separates into different subtrees
    current = root
    while current:
        if p.val < current.val and q.val < current.val:
            current = current.left
        elif p.val > current.val and q.val > current.val:
            current = current.right
        else:
            return current


# Binary Tree Right Side View
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root: return []  # edge case
    # we do a bfs and pick the node at the rightmost part of each levels
    queue = [root]
    result = []
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(node.val)
    return result


# Validate Binary Search Tree
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    # This will be done with dfs assumming an initial boundary of -inf and +inf for the root of the BST
    def dfs(node, left, right):
        if not node:
            return True  # base case
        if not (left < node.val < right):
            return False
        # check left and right subtree recursively
        return dfs(node.left, left, node.val) and dfs(node.right, node.val, right)

    return dfs(root, float("-inf"), float("inf"))


# Kth smallest element in a BST
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    # we append every value in the left subtree to a stack and pop until we get to k
    # if we get pop all the values out and we're not at k yet, we append all the values, we switch to the right subtree
    # we use the dfs inorder traversal using a stack
    n, stack, current = 0, [], root
    while current or stack:
        while current:
            stack.append(current.val)
            current = current.left
        node = stack.pop()
        n += 1
        if n == k:
            return node
        current = node.right


# Same Tree
def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    # we compare the nodes in the two tree recursively
    if not p and not q:
        return True
    if p and q and p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    else:
        return False


# Sub Tree of another Tree
def isSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    # keep comparing the nodes with subroot to see if they are the same
    if not root: return False
    if not subRoot: return True  # a null tree is a subtree of a tree

    if isSameTree(root, subRoot):
        return True
    else:
        return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)


# Balanced Binary Tree
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    # we go bottom up recursively with depth first search

    def dfs(root):
        if not root: return [True, 0]
        left, right = dfs(root.left), dfs(root.right)
        balanced = left[0] and right[0] and abs(left[1] - right[1])
        return True  # would come back to this

    return dfs(root)


# Construct Binary Tree from Inorder and PreOrder Traversal
def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    # build a hashmap to map the numbers in inorder list to their index
    hashmap = {value: index for index, value in enumerate(inorder)}
    preorder_root = 0  # every value in the pre order can be considered to be a root

    # and this can be use to our advantage to recursively split the inorder to left and right subtrees
    def arraytoTree(left, right):  # will take in range of index for the inorder
        nonlocal preorder_root
        if left > right:
            return None
        root = TreeNode(preorder[preorder_root])
        root.left = arraytoTree(left, hashmap[root] - 1)
        root.right = arraytoTree(hashmap[root] + 1, right)
        preorder_root += 1
        return root

    return arraytoTree(0, len(inorder) - 1)


def goodNodes(self, root: TreeNode) -> int:
    # do a preodrder dfs traversal
    # maintain the maxVal as we go down the tree
    def dfs(node, maxVal):
        if not node:
            return 0
        res = 1 if node.val >= maxVal else 0
        maxVal = max(node.val, maxVal)  # updates the maximum value
        res += dfs(node.left, maxVal)
        res += dfs(node.right, maxVal)
        return res

    return dfs(root, root.val)


def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
    adj = {i: [] for i in range(n)}

    # Build up an adjancecy list from the edges list
    for nodeA, nodeB in edges:
        adj[nodeA].append(nodeB)
        adj[nodeB].append(nodeA)

    def dfs(curr, par):
        total_time = 0
        # we are adding the parent node to each recursively to avoid infinte loop
        for child in adj[curr]:
            if child == par:
                continue
            childTime = dfs(child, curr)
            if childTime or hasApple[child]:
                total_time += 2 + childTime
        return total_time

    return dfs(0, -1)


def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
    # omo this one choke, I'll need to revisit this later
    answer = [0 for _ in range(n)]
    adjList = [[] for _ in range(n)]

    # Build up and adjacency list
    for i in range(n):
        edge = edges[i]
        adjList[edge[0]].append(edge[1])
        adjList[edge[1]].append(edge[0])

    def dfs(node):
        answer[node] = 1  # this is just a way to mark this as visited
        # builds up the count for the current subtree recursively
        alphabelt = "abcdefghijklmnopqrstuvwxyz"
        hashmap = {i: 0 for i in alphabelt}
        hashmap[labels[node]] = 1
        for neighbour in hashmap[node]:
            if answer[node] != 1:
                temp = dfs(neighbour)
                for letter, count in temp.keys(): hashmap[letter] += count
        answer[node] = count[labels[node]]
        return count

    dfs(0)
    return answer


def longestPath(self, parent: List[int], s: str) -> int:
    # build up a node from the parent list
    children: List[List[int]] = [[] for _ in range(len(parent))]
    for child in range(1, len(parent)):
        # started from 1 since the root has no parent
        children[parent[child]].append(child)
    max_path = 0

    def dfs(node) -> int:
        nonlocal max_path
        # create a temp max and second max for each node recursively
        max_first, max_second = 0, 0
        for child in children[node]:
            if s[child] != s[node]:  # we want to ensure they don't have the same label
                current_length = dfs(child)
                if current_length > max_first:
                    max_first, max_second = current_length, max_first
                elif current_length > max_second:
                    max_second = current_length
        max_path = max(max_path, max_first + max_second + 1)
        return max_first + 1

    dfs(0)
    return max_path