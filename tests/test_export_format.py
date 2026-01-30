import re
import unittest

from bvh import Bvh


class TestExportFormat(unittest.TestCase):
    def test_frames_have_three_decimal_places(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        raw = mocap.raw_data
        lines = [l.strip() for l in raw.strip().splitlines()]
        # find MOTION section
        idx = lines.index("MOTION")
        # frames start after two header lines: Frames: and Frame Time:
        frame_lines = lines[idx + 3 :]
        self.assertTrue(len(frame_lines) >= 1)

        pattern = re.compile(r"^-?\d+\.\d{3}(\t-?\d+\.\d{3})*$")
        # check a few frames
        for i in range(min(3, len(frame_lines))):
            self.assertRegex(frame_lines[i], pattern)

    def test_frame_columns_preserved(self):
        mocap = Bvh.from_file("tests/test_mocapbank.bvh")
        raw = mocap.raw_data
        lines = [l.strip() for l in raw.strip().splitlines()]
        idx = lines.index("MOTION")
        first_frame = lines[idx + 3]
        values = re.split(r"\s+", first_frame)
        self.assertEqual(len(values), len(mocap.frames[0]))


if __name__ == "__main__":
    unittest.main()
