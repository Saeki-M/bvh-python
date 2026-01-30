import copy
import re


class BvhNode:
    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1 :]
        raise IndexError("key {} not found".format(key))

    def __repr__(self):
        return str(" ".join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:
    def __init__(self, data):
        self.root = BvhNode()
        self.frames = self.tokenize(data)
        # accessor for subscription-style joint/channel access
        self.joint = Bvh.JointAccessor(self)

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            mocap = cls(f.read())
        return mocap

    def __len__(self):
        """Return the length of the animation in milliseconds"""
        return round(self.nframes * self.frame_time * 1000)

    def tokenize(self, data):
        lines = re.split("\n|\r", data)
        first_round = [re.split("\\s+", line.strip()) for line in lines[:-1]]
        node_stack = [self.root]
        node = None
        data_start_idx = 0
        for line, item in enumerate(first_round):
            key = item[0]
            if key == "{":
                node_stack.append(node)
            elif key == "}":
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == "Frame" and item[1] == "Time:":
                data_start_idx = line
                break
        return [
            [float(scalar) for scalar in line]
            for line in first_round[data_start_idx + 1 :]
        ]

    def __getitem__(self, x):
        if isinstance(x, int):
            frames = self.frames[[round(x / (1000 * self.frame_time))]]
        elif isinstance(x, slice):
            start_time = x.start if x.start is not None else 0
            end_time = x.stop if x.stop is not None else -1

            start_frame = round(start_time / (1000 * self.frame_time))
            end_frame = round(end_time / (1000 * self.frame_time))
            frames = self.frames[start_frame : end_frame : x.step]
        else:
            raise KeyError

        new_bvh = copy.deepcopy(self)
        new_bvh.frames = frames
        return new_bvh

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter("JOINT"):
                iterate_joints(child)

        iterate_joints(next(self.root.filter("ROOT")))
        return joints

    # Use `bvh.joint.keys()` to list joint names instead of `get_joints_names()`

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter("JOINT")]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search("ROOT", name)
        if not found:
            found = self.search("JOINT", name)
        if found:
            return found[0]
        raise LookupError("joint not found")

    def joint_offset(self, name):
        joint = self.get_joint(name)
        offset = joint["OFFSET"]
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        joint = self.get_joint(name)
        return joint["CHANNELS"][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint["CHANNELS"][0])
        raise LookupError("joint not found")

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(self.frames[frame_index][joint_index + channel_index])
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def set_frame_joint_channel(self, frame_index, joint, channel, value):
        """Set a single channel value for a given frame and joint."""
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1:
            raise LookupError("channel not found")
        self.frames[frame_index][joint_index + channel_index] = float(value)

    def set_frame_joint_channels(self, frame_index, joint, channels, values):
        """Set multiple channel values for a given frame and joint.

        `values` can be a single scalar (which will be broadcast to all channels)
        or an iterable of the same length as `channels`.
        """
        if isinstance(values, (int, float)):
            values = [values] * len(channels)
        if len(values) != len(channels):
            raise ValueError("values must match channels length")
        joint_index = self.get_joint_channels_index(joint)
        for channel, val in zip(channels, values):
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1:
                raise LookupError("channel not found")
            self.frames[frame_index][joint_index + channel_index] = float(val)

    def set_frames_joint_channels(self, joint, channels, values):
        """Set channel values across all frames for a joint.

        `values` may be:
        - a single scalar -> set every specified channel to that scalar in every frame
        - a list of scalars of length `nframes` when `len(channels) == 1`
        - a list of lists with shape (nframes, len(channels))
        """
        n = self.nframes
        per_frame = []
        # single scalar
        if isinstance(values, (int, float)):
            per_frame = [[float(values)] * len(channels) for _ in range(n)]
        # list of scalars for single channel
        elif all(isinstance(v, (int, float)) for v in values) and len(channels) == 1:
            if len(values) != n:
                raise ValueError("values length must match number of frames")
            per_frame = [[float(v)] for v in values]
        else:
            # assume list of lists
            if len(values) != n:
                raise ValueError("values length must match number of frames")
            for v in values:
                if len(v) != len(channels):
                    raise ValueError("each frame entry must match channels length")
                per_frame.append([float(x) for x in v])

        for i, frame_vals in enumerate(per_frame):
            self.set_frame_joint_channels(i, joint, channels, frame_vals)

    # Proxy helpers for subscription-based access -------------------------------------------------
    class JointAccessor:
        """Accessor available as `bvh.joint` which allows `bvh.joint[joint_name]` access.

        Also provides mapping-like helpers:
          - `bvh.joint.keys()` -> list of joint names
          - `list(bvh.joint)` -> list of joint names
          - `name in bvh.joint` -> membership test
          - `len(bvh.joint)` -> number of joints
        """

        def __init__(self, bvh):
            self._bvh = bvh

        def __getitem__(self, joint_name):
            return Bvh.JointProxy(self._bvh, joint_name)

        def keys(self):
            """Return a list of joint names in traversal order."""
            return [j.value[1] for j in self._bvh.get_joints()]

        def __iter__(self):
            yield from self.keys()

        def __contains__(self, name):
            return name in self.keys()

        def __len__(self):
            return len(self.keys())

    class JointProxy:
        """Proxy object returned by `bvh.joint[joint_name]`.

        Supports `joint_proxy[channel]` to get a `ChannelProxy` and
        `joint_proxy[channel] = value` to set the channel across all frames.
        """

        def __init__(self, bvh, joint_name):
            self._bvh = bvh
            self._joint = joint_name

        def keys(self):
            """Return channel names for this joint in order."""
            return self._bvh.joint_channels(self._joint)

        def __iter__(self):
            yield from self.keys()

        def __contains__(self, name):
            return name in self.keys()

        def __len__(self):
            return len(self.keys())

        def __getitem__(self, channel):
            # verify channel exists
            if self._bvh.get_joint_channel_index(self._joint, channel) == -1:
                raise LookupError("channel not found")
            return Bvh.ChannelProxy(self._bvh, self._joint, channel)

        def __setitem__(self, channel, value):
            # verify channel exists
            if self._bvh.get_joint_channel_index(self._joint, channel) == -1:
                raise LookupError("channel not found")
            self._bvh.set_frames_joint_channels(self._joint, [channel], value)

    class ChannelProxy:
        """Proxy object representing a specific channel of a joint.

        Supports indexing to get/set frame-specific values, e.g.:
            bvh.joint[joint][channel][frame]
            bvh.joint[joint][channel][slice]
        """

        def __init__(self, bvh, joint, channel):
            self._bvh = bvh
            self._joint = joint
            self._channel = channel

        def __repr__(self):
            vals = [
                self._bvh.frame_joint_channel(i, self._joint, self._channel)
                for i in range(self._bvh.nframes)
            ]
            return repr(vals)

        def __str__(self):
            vals = [
                self._bvh.frame_joint_channel(i, self._joint, self._channel)
                for i in range(self._bvh.nframes)
            ]
            return str(vals)

        def __getitem__(self, index):
            if isinstance(index, int):
                return self._bvh.frame_joint_channel(index, self._joint, self._channel)
            if isinstance(index, slice):
                return [
                    self._bvh.frame_joint_channel(i, self._joint, self._channel)
                    for i in range(*index.indices(self._bvh.nframes))
                ]
            raise TypeError("index must be int or slice")

        def __setitem__(self, index, value):
            if isinstance(index, int):
                self._bvh.set_frame_joint_channel(
                    index, self._joint, self._channel, value
                )
                return
            if isinstance(index, slice):
                indices = list(range(*index.indices(self._bvh.nframes)))
                if isinstance(value, (int, float)):
                    for i in indices:
                        self._bvh.set_frame_joint_channel(
                            i, self._joint, self._channel, value
                        )
                    return
                if len(value) != len(indices):
                    raise ValueError("values length must match slice length")
                for i, v in zip(indices, value):
                    self._bvh.set_frame_joint_channel(i, self._joint, self._channel, v)
                return
            raise TypeError("index must be int or slice")

    # end proxy helpers -------------------------------------------------------------------------

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        return len(self.frames)

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter("Frame")).value[2])
        except StopIteration:
            raise LookupError("frame time not found")

    @property
    def frame_rate(self):
        return 1 / self.frame_time

    @property
    def raw_data(self):
        _, root, _, _, _ = self.root
        data = "HIERARCHY\n"

        data, depth = self.write_node(root, data, 0)

        data += "MOTION\n"
        data += f"Frames:\t{self.nframes}\n"
        data += f"Frame Time:\t{self.frame_time}\n"

        for frame in self.frames:
            # ensure frame channel values are written with 3 decimal places
            data += "\t".join(format(f, ".3f") for f in frame) + "\n"

        return data

    def write_node(self, node, data, depth):
        n_type = node.value[0]

        data += "\t" * depth + "\t".join(node.value) + "\n"
        data += "\t" * depth + "{\n"
        data += "\t" * (depth + 1) + "\t".join(node.children[0].value) + "\n"
        if n_type != "End":
            data += "\t" * (depth + 1) + "\t".join(node.children[1].value) + "\n"
        for child in node.children[2:]:
            depth += 1
            data, depth = self.write_node(child, data, depth)
        data += "\t" * depth + "}\n"
        depth -= 1
        return data, depth

    def export(self, file):
        with open(file, "w") as f:
            f.write(self.raw_data)
