"""完整实例上的手工调权、学习策略校准和外层留出诊断。"""

from itertools import combinations, product
from pathlib import Path

import numpy as np

from .learning_dataset import assert_disjoint, grouped_folds
from .learning_model import DeltaModel
from .reporting import aggregate_pairs, pair_row, write_csv
from .settings import EvaluationOptions, SelectionOptions
from .storage import file_fingerprint, save_json


HAND_WEIGHTS = (1., 1.5, 2., 2.5, 3., 5., 10.)
LEARNING_METHODS = ('handcrafted_calibrated', 'size_handcrafted', 'learned_ridge', 'learned_tree')


def predict_instances(model, instances):
    """输入回归器和完整实例，返回以实例身份索引的候选差值预测，保持原候选顺序。"""
    return {i['id']: model.predict(i['candidates']) for i in instances}


def choose_pairs(instances, method, options, predictions=None):
    """输入实例、冻结参数和可选预测，仅根据求解前评分选候选，再关联标签形成评价行。"""
    pairs = []
    for instance in instances:
        rows = instance['candidates']
        if options.force_stay or method == 'symmetric_mst':
            chosen = 0
        elif method.startswith('learned_'):
            values = predictions[instance['id']]
            chosen = int(np.argmax(values[:, 1] - options.cost_weight * values[:, 0]))
        else:
            weight = (options.size_weights[str(instance['size'])] if method == 'size_handcrafted'
                      else options.geometry_weight)
            scores = []
            for row in rows:
                f = row['features']
                scores.append(-f['delta_binary_sum'] / max(f['before_binary_sum'], 1) -
                              weight * f['delta_geometry_sum'] / max(f['before_geometry_sum'], 1e-12))
            chosen = int(np.argmax(scores))
        pairs.append(dict(pair_row(rows[0], rows[chosen]), instance_id=instance['id'],
                          family_id=instance['family_id'], method=method))
    return pairs


def calibrate(instances, method, predictions=None, budgets=(.08, .09, .10), seed=0):
    """在校准实例上比较有限权重，以总体成本约束下的最高节时选择参数，返回各预算配置。"""
    if method == 'size_handcrafted':
        sizes = sorted({i['size'] for i in instances})
        choices = [SelectionOptions(size_weights=dict(zip(map(str, sizes), weights)), seed=seed)
                   for weights in product(HAND_WEIGHTS, repeat=len(sizes))]
    elif method.startswith('learned_'):
        choices = [SelectionOptions(cost_weight=float(w), seed=seed) for w in np.r_[0., np.logspace(-3, 3, 49)]]
    else:
        choices = [SelectionOptions(geometry_weight=w, seed=seed) for w in HAND_WEIGHTS]
    # 保留全体 stay，使校准在没有合适修复时有明确且可部署的退路。
    choices.append(SelectionOptions(force_stay=True, seed=seed))
    best = {}
    for options in choices:
        pairs = choose_pairs(instances, method, options, predictions)
        cost = sum(p['cost'] for p in pairs) / sum(p['baseline_cost'] for p in pairs) - 1
        saving = 1 - sum(p['time'] for p in pairs) / sum(p['baseline_time'] for p in pairs)
        rank = (saving, -cost)
        for budget in budgets:
            key = str(float(budget))
            if cost <= budget + 1e-12 and (key not in best or rank > best[key]['rank']):
                best[key] = {'rank': rank, 'options': options.to_dict(), 'cost_change': cost,
                             'phase2_saving': saving, 'calibration_cost_limit': budget}
    return {key: {k: v for k, v in value.items() if k != 'rank'} for key, value in best.items()}


def cross_predictions(instances, kind, folds=3, seed=0):
    """在完整实例内层交叉拟合，输出每个实例由未见过其标签的模型产生的预测。"""
    predictions = {}
    for index, (train, heldout) in enumerate(grouped_folds(instances, folds, seed)):
        assert_disjoint(train, heldout)
        model = DeltaModel(kind, seed + index).fit(train)
        predictions.update(predict_instances(model, heldout))
    return predictions


def prediction_diagnostics(instances, predictions):
    """比较同实例内的成本/时间符号与排序，输出辅助诊断；排除人为固定为零的 stay。"""
    actual, predicted, order = [], [], [[], []]
    for instance in instances:
        rows = instance['candidates'][1:]
        truth = np.asarray([[r['delta_cost'], r['saved_phase2_seconds']] for r in rows])
        estimate = predictions[instance['id']][1:]
        actual.extend(truth)
        predicted.extend(estimate)
        for first, second in combinations(range(len(rows)), 2):
            for column in range(2):
                delta = truth[first, column] - truth[second, column]
                if abs(delta) > 1e-10:
                    guess = estimate[first, column] - estimate[second, column]
                    order[column].append(.5 if abs(guess) <= 1e-10 else float(delta * guess > 0))
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {name: {'mae': float(np.mean(abs(actual[:, c] - predicted[:, c]))),
                   'sign_accuracy': float(np.mean(np.sign(actual[:, c]) == np.sign(predicted[:, c]))),
                   'within_instance_order_accuracy': float(np.mean(order[c])) if order[c] else None}
            for c, name in enumerate(('delta_cost', 'saved_phase2_seconds'))}


def summarize_method(pairs, options, bootstrap):
    """输入所选候选的完整实例配对行，返回总体与逐规模统计，标明使用离线标签。"""
    result = aggregate_pairs(pairs, bootstrap, options=options)
    result['by_size'] = {str(size): aggregate_pairs([p for p in pairs if p['size'] == size], bootstrap, options=options)
                         for size in sorted({p['size'] for p in pairs})}
    result['timing_kind'] = 'offline_group_observations'
    result['not_an_online_speed_measurement'] = True
    return result


def write_learning_report(output, report, pairs):
    """输入学习诊断统计及逐实例记录，输出 JSON、CSV 和中文摘要。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'learning_report.json', report)
    write_csv(output / 'learning_per_instance.csv', pairs)
    lines = ['# 客户划分选择开发报告', '',
             '以下使用离线候选标签，不含新策略的实际在线计时；不能替代独立测试。', '',
             '| 方法 | 成本变化 | 第二阶段节时 | 成本超限比例 | 最坏成本增加 |',
             '|---|---:|---:|---:|---:|']
    for method, stats in report['methods'].items():
        lines.append(f'| {method} | {stats["cost_change"]:+.2%} | {stats["phase2_saving"]:.2%} | '
                     f'{stats["cost_over_limit_fraction"]:.1%} | {stats["worst_cost_change"]:+.2%} |')
    lines += ['', '配置、实例隔离、8%/9%/10% 校准曲线及预测诊断见 learning_report.json。',
              '按规模汇总和逐实例选择见 JSON 与 CSV；同一实例的候选不跨训练/留出边界。']
    (output / 'learning_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def diagnose(dataset, output, folds=5, inner_folds=3, seed=0, options=None, bootstrap=1000):
    """输入开发数据，执行外层留出、内层校准；输出各方法的外层选择表现，不导出正式策略。"""
    options = options or EvaluationOptions()
    instances = dataset['instances']
    budgets = sorted({.08, .09, .10, options.cost_limit})
    all_pairs = {m: [] for m in ('symmetric_mst', 'handcrafted') + LEARNING_METHODS}
    report = {'kind': 'nested_grouped_development_diagnosis', 'thresholds': options.to_dict(),
              'sources': dataset['sources'], 'folds': [], 'methods': {}, 'seed': seed}
    for index, (train, heldout) in enumerate(grouped_folds(instances, folds, seed)):
        print(f'开发诊断 {index + 1}/{folds}：训练 {len(train)}、留出 {len(heldout)} 个完整实例。', flush=True)
        assert_disjoint(train, heldout)
        fold = {'index': index, 'train_families': [i['family_id'] for i in train],
                'heldout_families': [i['family_id'] for i in heldout], 'calibration': {}, 'predictions': {}}
        for method in ('symmetric_mst', 'handcrafted'):
            all_pairs[method].extend(choose_pairs(heldout, method, SelectionOptions()))
        for method in LEARNING_METHODS:
            train_predictions = heldout_predictions = None
            if method.startswith('learned_'):
                kind = method.removeprefix('learned_')
                train_predictions = cross_predictions(train, kind, inner_folds, seed + index)
                model = DeltaModel(kind, seed + index).fit(train)
                heldout_predictions = predict_instances(model, heldout)
                fold['predictions'][method] = prediction_diagnostics(heldout, heldout_predictions)
            curve = calibrate(train, method, train_predictions, budgets, seed)
            selection = SelectionOptions(**curve[str(float(options.cost_limit))]['options'])
            fold['calibration'][method] = curve
            all_pairs[method].extend(choose_pairs(heldout, method, selection, heldout_predictions))
        report['folds'].append(fold)
    for method, pairs in all_pairs.items():
        report['methods'][method] = summarize_method(pairs, options, bootstrap)
    write_learning_report(output, report, sum(all_pairs.values(), []))
    return report


def fit_bundle(training, validation, output, seed=0, options=None, bootstrap=1000):
    """在训练实例上拟合、在独立验证实例上校准，保存冻结策略及模型，返回验证开发报告。"""
    options = options or EvaluationOptions()
    train, valid = training['instances'], validation['instances']
    assert_disjoint(train, valid)
    if training['contract'] != validation['contract']:
        raise ValueError('训练和验证标签的求解环境或候选配置不一致。')
    output = Path(output)
    if (output / 'policy.json').exists():
        raise ValueError('策略目录已存在，请使用新的输出目录保存本次冻结参数。')
    output.mkdir(parents=True, exist_ok=True)
    budgets = sorted({.08, .09, .10, options.cost_limit})
    bundle = {'kind': 'frozen_partition_policy', 'evaluation': options.to_dict(),
              'contract': training['contract'], 'feature_names': training['feature_names'],
              'training_sources': training['sources'], 'validation_sources': validation['sources'],
              'training_families': [i['family_id'] for i in train],
              'calibration_families': [i['family_id'] for i in valid], 'methods': {}, 'seed': seed}
    report = {'kind': 'calibrated_validation_development', 'thresholds': options.to_dict(),
              'calibration': {}, 'prediction_diagnostics': {}, 'methods': {}}
    all_pairs = []
    for method in ('symmetric_mst', 'handcrafted') + LEARNING_METHODS:
        predictions, model_info = None, {}
        if method.startswith('learned_'):
            print(f'拟合 {method}：训练 {len(train)}、校准 {len(valid)} 个实例。', flush=True)
            model = DeltaModel(method.removeprefix('learned_'), seed).fit(train, training['sources'])
            predictions = predict_instances(model, valid)
            model_path = output / (method + '.joblib')
            model.save(model_path)
            save_json(output / (method + '_metadata.json'), model.metadata)
            model_info = {'model_file': model_path.name, 'model_fingerprint': file_fingerprint(model_path)}
            report['prediction_diagnostics'][method] = prediction_diagnostics(valid, predictions)
        if method in LEARNING_METHODS:
            curve = calibrate(valid, method, predictions, budgets, seed)
            report['calibration'][method] = curve
            selected = curve[str(float(options.cost_limit))]['options']
        else:
            selected = SelectionOptions(seed=seed).to_dict()
        pairs = choose_pairs(valid, method, SelectionOptions(**selected), predictions)
        all_pairs.extend(pairs)
        report['methods'][method] = summarize_method(pairs, options, bootstrap)
        bundle['methods'][method] = dict(model_info, options=selected)
    bundle['methods']['random'] = {'options': SelectionOptions(seed=seed).to_dict()}
    save_json(output / 'policy.json', bundle)
    write_learning_report(output, report, all_pairs)
    return report
