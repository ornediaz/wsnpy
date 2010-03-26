class Node:
    def __init__(z, x):
        z.x = x
    def add(z, x):
        if x < z.x:
            if hasattr(z, 'left'):
                z.left.add(x)
            else:
                z.left = Node(x)
        if x > z.x:
            if hasattr(z, 'right'):
                z.right.add(x)
            else:
                z.right = Node(x)
    def preorder(z):
        l = [z]
        while l:
            n = l.pop(-1)
            print n.x
            if hasattr(n, 'right'):
                l.append(n.right)
            if hasattr(n, 'left'):
                l.append(n.left)
    def postorder(z):
        l = [z]
        depth = 0
        while l:
            n = l[-1]
            if not hasattr(n, 'left'):
                print n.x
                depth += 1
                l.pop(-1)
            elif depth > 1:
                print n.x
                depth -= 1
                l.pop(-1)
            else:
                l.append(n.right)
                l.append(n.left)
n = Node(5)
for d in (3, 1, 4, 8, 7, 10):
    n.add(d)
n.postorder()
