from kdtree import KDTree
import unittest


class TestKDTree(unittest.TestCase):

    def test_kdtree(self):

        tree = KDTree()
        tree.insert((2, 6))
        tree.insert((3, 1))
        tree.insert((8, 7))
        tree.insert((10, 2))
        tree.insert((13, 3))

        assert tree.get_nearest((9, 4)) == ((10, 2), 2.23606797749979)
        assert tree.get_nearest((4, 1.5))[0] == (3, 1)
        assert tree.get_nearest((7, 8))[0] == (8, 7)
        assert tree.get_nearest((11, 1))[0] == (10, 2)
        assert tree.get_nearest((13, 3))[0] == (13, 3)


if __name__ == '__main__':
    unittest.main()
