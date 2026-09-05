"""小型 GraphML 上的采集、续跑、实测和报告整条命令行集成测试。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from src.partition_repair.evaluator import evaluate_group
from src.partition_repair.reporting import candidate_report, evaluation_report
from src.partition_repair.runner import build_parser, collect, evaluate
from src.partition_repair.storage import read_json
from tests.partition_repair_fixtures import tiny_model


class PipelineTests(unittest.TestCase):
    """验证服务器脚本依赖的真实文件格式与断点续跑，不加载真实规模地图。"""

    def test_collect_resume_evaluate_and_report(self):
        """生成小地图，真实采集并复测；续跑不重复已完成组，复测明确执行新求解。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_path = root / 'tiny.graphml'
            graph = tiny_model(customer_count=7, depot_count=2).graph.copy()
            for node in graph:
                longitude, latitude = graph.nodes[node].pop('pos')
                graph.nodes[node].update(x=longitude, y=latitude)
            nx.write_graphml(graph, graph_path)
            collect_dir = root / 'collect'
            arguments = ['--output', str(collect_dir), '--graph', str(graph_path), '--stage', 'B',
                         '--sizes', '4', '--instances-per-size', '1', '--depots', '2', '--drones', '2',
                         '--drone-limit', '.1', '--time-limit', '2']
            collect_args = build_parser('collect').parse_args(arguments)
            with redirect_stdout(io.StringIO()):
                collect(collect_args)
            groups = list((collect_dir / 'group_runs').glob('*.json'))
            self.assertTrue(groups)
            for path in groups:
                self.assertTrue(read_json(path)['complete'])
            with patch('src.partition_repair.runner.evaluate_group', side_effect=AssertionError('已完成组不应重算')):
                with redirect_stdout(io.StringIO()):
                    collect(collect_args)
            candidate_stats = candidate_report(collect_dir, collect_dir / 'report', bootstrap=10)
            self.assertTrue(candidate_stats['complete'])
            self.assertEqual(candidate_stats['complete_instances'], 1)
            eval_dir = root / 'evaluation'
            eval_args = build_parser('evaluate').parse_args([
                '--output', str(eval_dir), '--graph', str(graph_path), '--manifest', str(collect_dir / 'manifest.json'),
                '--methods', 'symmetric_mst', 'handcrafted', '--repeats', '1',
            ])
            with redirect_stdout(io.StringIO()):
                evaluate(eval_args)
            runs = read_json(eval_dir / 'evaluation_runs.json')
            self.assertEqual(len(runs), 2)
            self.assertTrue(all(not row['solver_result_cache_enabled'] for row in runs))
            self.assertTrue(all(row['online_seconds'] >= row['phase2_wall_seconds'] for row in runs))
            report = evaluation_report(eval_dir, eval_dir / 'report', bootstrap=10)
            self.assertTrue(report['methods']['handcrafted']['complete'])

    def test_interrupted_collection_resumes_from_completed_groups(self):
        """在第二个组故意中断，确认未完成标志保留，续跑仅重算缺失组并补齐实例。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_path = root / 'tiny.graphml'
            graph = tiny_model(customer_count=5, depot_count=2).graph.copy()
            for node in graph:
                longitude, latitude = graph.nodes[node].pop('pos')
                graph.nodes[node].update(x=longitude, y=latitude)
            nx.write_graphml(graph, graph_path)
            args = build_parser('collect').parse_args([
                '--output', str(root / 'collect'), '--graph', str(graph_path), '--stage', 'A',
                '--sizes', '4', '--instances-per-size', '1', '--depots', '2', '--drone-limit', '.1',
            ])
            count = 0

            def interrupt_second(*arguments, **keywords):
                """第一组真实求解后在第二组触发键盘中断，模拟可恢复的标签采集停止。"""
                nonlocal count
                count += 1
                if count == 2:
                    raise KeyboardInterrupt()
                return evaluate_group(*arguments, **keywords)

            with patch('src.partition_repair.runner.evaluate_group', side_effect=interrupt_second):
                with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                    collect(args)
            observations = read_json(root / 'collect' / 'group_runs.json')
            self.assertEqual(sum(row['complete'] for row in observations), 1)
            self.assertEqual(sum(not row['complete'] for row in observations), 1)
            with patch('src.partition_repair.runner.evaluate_group', wraps=evaluate_group) as resumed:
                with redirect_stdout(io.StringIO()):
                    collect(args)
                self.assertEqual(resumed.call_count, 1)
            self.assertTrue(read_json(root / 'collect' / 'instances.json')[0]['complete'])


if __name__ == '__main__':
    unittest.main()
