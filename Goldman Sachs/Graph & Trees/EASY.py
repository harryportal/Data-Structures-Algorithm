import math
from typing import Optional
from Learning.Trees import TreeNode



# 25 Invert Tree
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    # start with root, invert the child nodes, do the same recursively for the child nodes
    # O(n) runtime going through all the nodes and 0(n) for the recursive function call stack
    if not root: return None  # edge case
    root.left, root.right = root.right, root.left
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root


# 26 Binary Tree Inorder Traversal
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    # inorder traversal --> left -- root -- right
    # let's solve this with an iterative dfs using stack
    stack, result, curr = [], [], root
    if not curr: return []  # edge case
    while curr or stack:
        while curr:  # keep going along the left branch of the current node
            stack.append(curr.val)
            curr = curr.left
        node = stack.pop() # this pops the left most node in the first iteration
        result.append(node.val)
        if node.right: curr = node.right
    return result

# 43 Closest Binary Search Tree Value
def closestValue(self, root: Optional[TreeNode], target: float) -> int:
    # we can do a binary search since this is a binary search tree
    closest = math.inf
    while root:
        closest = min(root.val, closest, key=lambda x: abs(target - x))
        root = root.left if target < root.val else root.right
    return closest

# 41 Maximum Depth of Binary Tree
def minDepth(self, root: Optional[TreeNode]) -> int:
    # we use bfs to traverse level by level and return the first node with no children
    queue = [[root, 1]]
    while queue:
        node, depth = queue.pop(0)
        if not node.left and not node.right:
            return depth
        if node.left: queue.append([node.left, depth + 1])
        if node.right: queue.append([node.right, depth + 1])

# 33 Symmetric Tree
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if not root: return True  # edge case

    # we basically define an helper function to check the nodes recursively
    def check(nodeA, nodeB):
        if not nodeA and not nodeB: return True # an empty tree is symmetric of course
        if not nodeA or not nodeB or nodeA.val != nodeB.val:
            return False
        return check(nodeA.left, nodeB.right) and check(nodeA.right, nodeB.left)

    return check(root.left, root.right)
