#  Copyright (c) 2021. Spectar LLC - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential

import numpy as np
import unittest

from py3dtiles.i3dm import I3dm


class TestI3DM(unittest.TestCase):
    def test_read(self):
        with open('../tree.i3dm', 'rb') as reader:
            arr = reader.read()

        bytes_arr = arr
        arr = np.frombuffer(arr, dtype=np.uint8)

        i3dm = I3dm.from_array(arr)

        arr = i3dm.to_array().tobytes()
        # arr = i3dm.body.glTF.to_array().tobytes()

        with open('../tree.glb', 'wb') as writer:
            writer.write(arr)