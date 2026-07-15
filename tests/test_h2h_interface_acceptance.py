"""阶段 6 距离接口兼容验收：类型、NumPy 编号、错误和禁止物化。"""

import tempfile
import unittest

import numpy as np

from distance_oracle import UnsupportedDistanceOperation, build_distance_provider
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_20_node_graph


class H2HInterfaceAcceptanceTests(unittest.TestCase):
    """确认 truck/drone 均保持原双下标协议但拒绝整矩阵操作。"""

    def test_public_distance_protocol_and_errors(self):
        """返回 Python float、兼容 NumPy 整数，并对未知节点和遍历给出明确异常。"""
        builder_path, library_path = ensure_native_built('release')
        graph = build_fixed_20_node_graph()
        with tempfile.TemporaryDirectory(prefix='h2h-interface-') as temporary:
            distance = build_distance_provider(
                graph,
                backend='h2h',
                dataset_name='interface-acceptance',
                index_dir=temporary,
                builder_path=str(builder_path),
                library_path=str(library_path),
            )
            try:
                truck_value = distance['truck'][np.int64(0)][np.int32(10)]
                drone_value = distance['drone'][np.int64(0)][np.int32(10)]
                self.assertIs(type(truck_value), float)
                self.assertIs(type(drone_value), float)
                with self.assertRaisesRegex(KeyError, '未知源节点'):
                    _ = distance['truck'][999][0]
                with self.assertRaisesRegex(KeyError, '未知目标节点'):
                    _ = distance['drone'][0][999]
                for operation in (
                    lambda: iter(distance['truck']),
                    lambda: distance['truck'].items(),
                    lambda: distance['truck'][0].values(),
                    lambda: len(distance['drone']),
                ):
                    with self.assertRaises(UnsupportedDistanceOperation):
                        operation()
            finally:
                distance['truck'].close()


if __name__ == '__main__':
    unittest.main()
