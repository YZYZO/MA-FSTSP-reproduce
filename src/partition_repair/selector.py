"""相同候选集合上的一次选择；真实求解标签不参与在线决策。"""

import random

from .candidates import repair_score
from .settings import SelectionOptions
from .storage import fingerprint


BASE_METHODS = ('symmetric_mst', 'handcrafted', 'count_only', 'burden_only', 'random', 'original_mst')
POLICY_METHODS = ('handcrafted_calibrated', 'size_handcrafted', 'learned_ridge', 'learned_tree')
METHODS = BASE_METHODS + POLICY_METHODS


def selection_seed(context, seed=0, repeat=0, identity=None):
    """输入完整实例、实验种子和重复编号，返回跨进程稳定的选择种子，不改求解器种子。"""
    payload = {'depots': list(context.depots), 'cities': list(context.cities),
               'drone': context.model.drone, 'speed': context.model.speed,
               'limit': context.model.limit, 'identity': identity}
    return int(fingerprint({'instance': payload, 'seed': seed, 'repeat': repeat})[:16], 16)


def select_candidate(context, baseline, candidates, method='handcrafted', geometry_weight=1.0, seed=0,
                     *, options=None, model=None, repeat=0, identity=None):
    """输入候选和固定手工设置，输出一个候选；并列时保留靠前者，默认含 stay。"""
    options = options or SelectionOptions(geometry_weight=geometry_weight, seed=seed)
    if method == 'symmetric_mst' or options.force_stay:
        return candidates[0]
    if method == 'random':
        return random.Random(selection_seed(context, options.seed, repeat, identity)).choice(candidates)
    if method in ('handcrafted', 'handcrafted_calibrated', 'size_handcrafted'):
        weight = options.geometry_weight
        if method == 'size_handcrafted':
            weight = options.size_weights[str(len(context.cities))]
        return max(candidates, key=lambda c: repair_score(context, baseline, c.partition, weight))
    if method.startswith('learned_'):
        if model is None:
            raise ValueError('学习选择需要加载已训练模型。')
        records = [{'name': c.name, 'kind': c.kind, 'strength': c.strength,
                    'features': context.difference(baseline, c.partition)} for c in candidates]
        predictions = model.predict(records)
        # 两列依次是配送成本差和第二阶段节时；stay 的两列由模型入口固定为零。
        scores = predictions[:, 1] - options.cost_weight * predictions[:, 0]
        return candidates[int(scores.argmax())]
    if method == 'burden_only':
        return min(candidates, key=lambda c: context.partition(c.partition)['binary_sum'])
    if method == 'count_only':
        return min(candidates, key=lambda c: sum(len(c.partition[d]) ** 2 for d in context.depots))
    raise ValueError(f'未知候选选择方法：{method}')
