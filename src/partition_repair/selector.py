"""相同候选集合上的一次选择；真实求解标签不参与在线决策。"""

import random

from .candidates import repair_score


METHODS = ('symmetric_mst', 'handcrafted', 'count_only', 'burden_only', 'random', 'original_mst')


def select_candidate(context, baseline, candidates, method='handcrafted', geometry_weight=1.0, seed=0):
    """输入候选和固定手工设置，输出一个候选；并列时保留靠前者，默认含 stay。"""
    if method == 'symmetric_mst':
        return candidates[0]
    if method == 'random':
        return random.Random(seed).choice(candidates)
    if method == 'handcrafted':
        return max(candidates, key=lambda c: repair_score(context, baseline, c.partition, geometry_weight))
    if method == 'burden_only':
        return min(candidates, key=lambda c: context.partition(c.partition)['binary_sum'])
    if method == 'count_only':
        return min(candidates, key=lambda c: sum(len(c.partition[d]) ** 2 for d in context.depots))
    raise ValueError(f'未知候选选择方法：{method}')
