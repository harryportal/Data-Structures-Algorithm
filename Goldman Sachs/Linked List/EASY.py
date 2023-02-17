# 11 Linked list Cycle
from typing import Optional

from Learning.Linkedlist import ListNode


def hasCycle(self, head: Optional[ListNode]) -> bool:
    # This can be easily solved using hashset to store the nodes as they are visited and return True if a
    # duplicate node is found(not a duplicate value as the different nodes might have the same value)
    # However the space complexity is o(n) in worst case and this can be optimized to constant time by using
    # flood cycle's algorithm to detect cycle by using a slow and fast pointer to fo through the linked list
    # we move the slow pointer once and the fast pointer twice, if there is a cycle these two would eventually
    # catch up
    fast = slow = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return True
    return False


# 24 Middle of Linked list
def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # One approach is to store all the value in the linked list and return the value at the middle
    # We can reduce the space complexity to 0(1) by using a slow and fast pointer
    # the fast pointer will traverse the linked list twice as fast as the slow pointer so when we fast pointer is at
    # the end, the slow will definitely point to the middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# 27 Reverse Linked list
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # reversing the linked list means we now want the tail to be the head, first off remember the tail points to None
    # if the head is going to be the tail now, it has to point to None and then we keep reversing the pointers
    if not head: return None
    prev, curr = None, head
    while curr:
        temp = curr.next  # temporarily store the next node in the linked list
        curr.next = prev  # make the current node point to the prev node(in this case starts from None)
        prev = curr
        curr = temp
    return prev


# 30 Delete Duplicates from sorted Lists
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    pointer = head
    while pointer and pointer.next:
        if pointer.val == pointer.next.val:
            pointer.next = pointer.next.next
        else:
            pointer = pointer.next
    return head
