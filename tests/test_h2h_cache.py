"""验证 H2H 图哈希、版本化缓存、失效和中断目录恢复。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h2h_backend import (
    INDEX_FORMAT_VERSION,
    compute_graph_hash,
    ensure_h2h_index,
    normalize_graph_for_h2h,
    validate_h2h_cache,
)
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_20_node_graph


class H2HCacheLifecycleTests(unittest.TestCase):
    """确认缓存只按完整图哈希命中，且不会加载不完整目录。"""

    @classmethod
    def setUpClass(cls):
        """编译一次 Release 原生后端并记录 builder 路径。"""
        cls.builder_path, _ = ensure_native_built('release')

    def test_first_build_then_second_call_hits_same_cache(self):
        """第一次调用必须构建，第二次必须只命中同一 READY 缓存。"""
        graph = build_fixed_20_node_graph()
        with tempfile.TemporaryDirectory(prefix='h2h-cache-hit-') as temp:
            first = ensure_h2h_index(
                graph, 'cache-hit', index_dir=temp, builder_path=self.builder_path
            )
            first_index_mtime = first.index_path.stat().st_mtime_ns
            second = ensure_h2h_index(
                graph, 'cache-hit', index_dir=temp, builder_path=self.builder_path
            )
            self.assertTrue(first.built)
            self.assertFalse(second.built)
            self.assertEqual(first.cache_dir, second.cache_dir)
            self.assertEqual(first_index_mtime, second.index_path.stat().st_mtime_ns)
            self.assertTrue(validate_h2h_cache(second.cache_dir, second.graph_hash))

            metadata = json.loads(second.metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(metadata['graph_hash'], second.graph_hash)
            self.assertEqual(metadata['node_count'], graph.number_of_nodes())
            self.assertEqual((second.cache_dir / 'READY').read_text().strip(), second.graph_hash)

    def test_coordinate_edge_weight_and_direction_change_hash(self):
        """坐标、任意边权或方向变化都必须产生不同 SHA-256。"""
        original = build_fixed_20_node_graph()
        baseline_hash = compute_graph_hash(normalize_graph_for_h2h(original))

        coordinate_changed = copy.deepcopy(original)
        coordinate_changed.nodes[0]['pos'][0] += 1e-7
        self.assertNotEqual(
            baseline_hash,
            compute_graph_hash(normalize_graph_for_h2h(coordinate_changed)),
        )

        weight_changed = copy.deepcopy(original)
        weight_changed.edges[0, 1, 0]['weight'] += 1e-6
        self.assertNotEqual(
            baseline_hash,
            compute_graph_hash(normalize_graph_for_h2h(weight_changed)),
        )

        direction_changed = copy.deepcopy(original)
        weight = direction_changed.edges[4, 9, 0]['weight']
        direction_changed.remove_edge(4, 9, 0)
        direction_changed.add_edge(9, 4, weight=weight)
        self.assertNotEqual(
            baseline_hash,
            compute_graph_hash(normalize_graph_for_h2h(direction_changed)),
        )

    def test_weight_change_builds_new_cache_without_overwriting_old(self):
        """修改任意最小边权后应建立新目录，旧缓存仍保持有效。"""
        graph = build_fixed_20_node_graph()
        changed = copy.deepcopy(graph)
        changed.edges[0, 1, 0]['weight'] *= 0.9
        with tempfile.TemporaryDirectory(prefix='h2h-cache-invalidate-') as temp:
            original_cache = ensure_h2h_index(
                graph, 'invalidate', index_dir=temp, builder_path=self.builder_path
            )
            changed_cache = ensure_h2h_index(
                changed, 'invalidate', index_dir=temp, builder_path=self.builder_path
            )
            self.assertTrue(original_cache.built)
            self.assertTrue(changed_cache.built)
            self.assertNotEqual(original_cache.graph_hash, changed_cache.graph_hash)
            self.assertNotEqual(original_cache.cache_dir, changed_cache.cache_dir)
            self.assertTrue(validate_h2h_cache(original_cache.cache_dir, original_cache.graph_hash))
            self.assertTrue(validate_h2h_cache(changed_cache.cache_dir, changed_cache.graph_hash))

    def test_invalid_final_is_quarantined_and_interrupted_build_is_ignored(self):
        """无 READY 最终目录不得加载；旧 `.building-*` 不阻止重新构建。"""
        graph = build_fixed_20_node_graph()
        graph_hash = compute_graph_hash(normalize_graph_for_h2h(graph))
        cache_name = f'recovery-{graph_hash}-h2h-v{INDEX_FORMAT_VERSION}'
        with tempfile.TemporaryDirectory(prefix='h2h-cache-recovery-') as temp:
            root = Path(temp)
            invalid_final = root / cache_name
            invalid_final.mkdir()
            (invalid_final / 'metadata.json').write_text('{}', encoding='utf-8')
            interrupted = root / f'{cache_name}.building-interrupted'
            interrupted.mkdir()
            (interrupted / 'partial').write_text('incomplete', encoding='utf-8')

            result = ensure_h2h_index(
                graph, 'recovery', index_dir=root, builder_path=self.builder_path
            )
            self.assertTrue(result.built)
            self.assertTrue(validate_h2h_cache(result.cache_dir, graph_hash))
            self.assertTrue(interrupted.is_dir())
            quarantined = list(root.glob(f'{cache_name}.invalid-*'))
            self.assertEqual(len(quarantined), 1)

    def test_truncated_graph_binary_invalidates_ready_cache(self):
        """即使 READY/索引存在，graph.bin 截断也必须隔离旧目录并完整重建。"""
        graph = build_fixed_20_node_graph()
        with tempfile.TemporaryDirectory(prefix='h2h-cache-truncated-graph-') as temp:
            first = ensure_h2h_index(
                graph, 'truncated', index_dir=temp, builder_path=self.builder_path
            )
            with first.graph_path.open('r+b') as graph_file:
                graph_file.truncate(first.graph_path.stat().st_size - 1)
            self.assertFalse(validate_h2h_cache(first.cache_dir, first.graph_hash))

            rebuilt = ensure_h2h_index(
                graph, 'truncated', index_dir=temp, builder_path=self.builder_path
            )
            self.assertTrue(rebuilt.built)
            self.assertTrue(validate_h2h_cache(rebuilt.cache_dir, rebuilt.graph_hash))
            quarantined = list(Path(temp).glob(f'{first.cache_dir.name}.invalid-*'))
            self.assertEqual(len(quarantined), 1)

    def test_auto_build_disabled_rejects_missing_cache(self):
        """禁用自动构建时，缓存缺失必须报错且不能创建临时构建目录。"""
        graph = build_fixed_20_node_graph()
        with tempfile.TemporaryDirectory(prefix='h2h-cache-disabled-') as temp:
            with patch('h2h_backend.H2H_AUTO_BUILD', False):
                with self.assertRaisesRegex(FileNotFoundError, '缓存缺失或无效'):
                    ensure_h2h_index(graph, 'disabled', index_dir=temp)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
