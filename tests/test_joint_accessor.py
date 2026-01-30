import unittest

from bvh import Bvh


class TestJointAccessor(unittest.TestCase):
    def test_keys_exist(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        keys = mocap.joint.keys()
        self.assertTrue(isinstance(keys, list))
        self.assertTrue(len(keys) > 0)
        self.assertEqual(keys[0], mocap.get_joints()[0].name)

    def test_iter_and_contains_and_len(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        names = list(mocap.joint)
        self.assertTrue(len(names) > 0)
        self.assertEqual(len(names), len(mocap.joint))
        self.assertTrue(names[0] in mocap.joint)

    def test_joint_proxy_keys_and_access(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        channels = mocap.joint["Head"].keys()
        self.assertEqual(channels, mocap.joint_channels("Head"))
        self.assertTrue("Yrotation" in mocap.joint["Head"])
        self.assertEqual(mocap.joint["Head"]["Yrotation"][22], 8.47)

    def test_channel_proxy_prints_values(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        vals = mocap.joint["Head"]["Yrotation"][:]
        self.assertEqual(repr(mocap.joint["Head"]["Yrotation"]), repr(vals))
        self.assertEqual(str(mocap.joint["Head"]["Yrotation"]), str(vals))


if __name__ == "__main__":
    unittest.main()
