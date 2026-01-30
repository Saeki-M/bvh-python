import unittest

from bvh import Bvh


class TestSetters(unittest.TestCase):
    def test_set_frame_channel(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        orig = mocap.frame_joint_channel(0, "Hips", "Xposition")
        mocap.set_frame_joint_channel(0, "Hips", "Xposition", 42.42)
        self.assertEqual(mocap.frame_joint_channel(0, "Hips", "Xposition"), 42.42)
        # restore to avoid side effects for other tests
        mocap.set_frame_joint_channel(0, "Hips", "Xposition", orig)

    def test_set_frame_channels(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        mocap.set_frame_joint_channels(
            1, "Head", ["Xrotation", "Yrotation"], [9.9, 8.8]
        )
        self.assertEqual(
            mocap.frame_joint_channels(1, "Head", ["Xrotation", "Yrotation"]),
            [9.9, 8.8],
        )

    def test_set_frames_single_channel(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        mocap.set_frames_joint_channels("Hips", ["Xposition"], 0.0)
        vals = [v[0] for v in mocap.frames_joint_channels("Hips", ["Xposition"])]
        self.assertEqual(len(vals), mocap.nframes)
        self.assertTrue(all(abs(x) < 1e-9 for x in vals))

    def test_set_unknown_channel_raises(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        with self.assertRaises(LookupError):
            mocap.set_frame_joint_channel(0, "Hips", "Badchannel", 1.0)

    def test_proxy_get_and_set(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        # get via proxies
        self.assertEqual(mocap.joint["Head"]["Yrotation"][22], 8.47)

        # set single frame via channel proxy
        orig = mocap.joint["Head"]["Yrotation"][22]
        mocap.joint["Head"]["Yrotation"][22] = 99.9
        self.assertEqual(mocap.joint["Head"]["Yrotation"][22], 99.9)
        mocap.joint["Head"]["Yrotation"][22] = orig

        # set across all frames via joint proxy
        mocap.joint["Hips"]["Xposition"] = 0.0
        vals = [v[0] for v in mocap.frames_joint_channels("Hips", ["Xposition"])]
        self.assertTrue(all(abs(x) < 1e-9 for x in vals))

    def test_proxy_unknown_channel_raises(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        with self.assertRaises(LookupError):
            mocap.joint["Hips"]["Badchannel"] = 1.0
        with self.assertRaises(LookupError):
            _ = mocap.joint["Hips"]["Badchannel"][0]


if __name__ == "__main__":
    unittest.main()
