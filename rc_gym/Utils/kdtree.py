import math


def closest_node(values, node1, node2):

    if node1 is None:
        return node2, node2.distance2_to(values) if node2 is not None else math.inf

    if node2 is None:
        return node1, node1.distance2_to(values) if node1 is not None else math.inf

    node1_dist2 = node1.distance2_to(values)
    node2_dist2 = node2.distance2_to(values)

    if node1_dist2 < node2_dist2:
        return node1, node1_dist2
    else:
        return node2, node2_dist2


class KDTree:
    class KDTreeNode:

        def __init__(self, values, left=None, right=None):
            self.values = values
            self.left = left
            self.right = right

        def insert(self, values, depth=0):
            if self.values is None:
                self.values = values
            else:
                if values[depth % len(values)] < self.values[depth % len(self.values)]:
                    if self.left is None:
                        self.left = KDTree.KDTreeNode(values)
                    else:
                        self.left.insert(values, depth+1)
                else:
                    if self.right is None:
                        self.right = KDTree.KDTreeNode(values)
                    else:
                        self.right.insert(values, depth+1)

        def distance2_to(self, values):
            d2 = 0
            for i in range(len(values)):
                d2 += (values[i] - self.values[i])**2

            return d2

        def get_nearest(self, values, depth=0):
            if self.values is None:
                return None, math.inf

            if self.left is None and self.right is None:
                return self, self.distance2_to(values)

            if values[depth % len(values)] < self.values[depth % len(self.values)]:
                next_branch = self.left
                other_branch = self.right
            else:
                next_branch = self.left
                other_branch = self.right

            if next_branch is not None:
                other, _ = next_branch.get_nearest(values, depth+1)
                closest, closest_dist2 = closest_node(values, other, self)
            else:
                closest, closest_dist2 = self, self.distance2_to(values)

            line_dist = values[depth % len(values)] - self.values[depth % len(self.values)]

            if other_branch is not None:
                if closest_dist2 >= line_dist**2:
                    other, _ = other_branch.get_nearest(values, depth+1)
                    closest, closest_dist2 = closest_node(values, other, closest)

            return closest, closest_dist2

    def __init__(self):
        self.root = KDTree.KDTreeNode(None)

    def insert(self, values):
        self.root.insert(values)

    def get_nearest(self, values):
        node, dist2 = self.root.get_nearest(values)
        return node.values, math.sqrt(dist2)
