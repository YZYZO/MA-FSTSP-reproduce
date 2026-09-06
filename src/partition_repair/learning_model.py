"""CPU 上的成本差和第二阶段节时回归；训练预处理只接触训练实例。"""

from importlib.metadata import version

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .learning_dataset import FEATURE_NAMES, feature_vector


class DeltaModel:
    """保存两个回归器；输入求解前候选记录，输出成本增量和完整第二阶段节时。"""

    def __init__(self, kind='tree', seed=0):
        """输入固定模型类型与种子，初始化少量参数；不使用候选行级随机早停。"""
        if kind not in ('ridge', 'tree'):
            raise ValueError('模型类型必须为 ridge 或 tree。')
        self.kind, self.seed = kind, seed
        self.feature_names = list(FEATURE_NAMES)
        self.models = []
        self.metadata = {}

    def fit(self, instances, sources=None):
        """输入完整训练实例及来源，等权对待实例，分别拟合两个差值标签，返回模型自身。"""
        records = [r for i in instances for r in i['candidates']]
        X = np.asarray([feature_vector(r) for r in records], dtype=float)
        targets = np.asarray([[r['delta_cost'], r['saved_phase2_seconds']] for r in records])
        # 每实例总权重相同，候选去重后数量较少的实例不会被低估。
        weights = np.asarray([1 / len(i['candidates']) for i in instances for _ in i['candidates']])
        weights *= len(weights) / weights.sum()
        self.models = []
        with threadpool_limits(limits=1):
            for column in range(2):
                if self.kind == 'ridge':
                    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                    model.fit(X, targets[:, column], ridge__sample_weight=weights)
                else:
                    model = HistGradientBoostingRegressor(max_iter=80, max_depth=2, max_leaf_nodes=4,
                                                         min_samples_leaf=10, l2_regularization=2.0,
                                                         learning_rate=.08, early_stopping=False,
                                                         random_state=self.seed)
                    model.fit(X, targets[:, column], sample_weight=weights)
                self.models.append(model)
        self.metadata = {'kind': self.kind, 'seed': self.seed, 'feature_names': self.feature_names,
                         'training_families': [i['family_id'] for i in instances],
                         'training_instance_ids': [i['id'] for i in instances], 'sources': sources or [],
                         'packages': {name: version(name) for name in ('scikit-learn', 'numpy', 'joblib', 'threadpoolctl')},
                         'parameters': [m.get_params(deep=False) if self.kind == 'tree' else
                                        {'standard_scaler': True, 'ridge_alpha': 10.0} for m in self.models]}
        return self

    def predict(self, records):
        """输入无需求解标签的候选记录，输出 n×2 差值矩阵；stay 的两项严格为零。"""
        if self.feature_names != list(FEATURE_NAMES):
            raise ValueError('模型特征版本与当前代码不一致。')
        X = np.asarray([feature_vector(r) for r in records], dtype=float)
        with threadpool_limits(limits=1):
            result = np.column_stack([m.predict(X) for m in self.models])
        for index, record in enumerate(records):
            if record['name'] == 'stay':
                result[index] = 0.0
        return result

    def save(self, path):
        """将训练后的模型保存到指定文件，供冻结策略重复加载。"""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """加载本项目生成的模型文件，并核对特征顺序与学习依赖版本。"""
        model = joblib.load(path)
        if model.feature_names != list(FEATURE_NAMES):
            raise ValueError('模型特征与当前代码不一致。')
        for name, expected in model.metadata['packages'].items():
            if version(name) != expected:
                raise ValueError(f'模型依赖 {name} 应为 {expected}，请使用相同学习环境。')
        return model
