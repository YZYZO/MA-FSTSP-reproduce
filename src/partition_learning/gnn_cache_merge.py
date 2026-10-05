"""把新增真实求解候选安全合并进已有候选划分GNN缓存。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _partition_signature(record: dict[str, Any]) -> tuple[int, ...]:
    """输入缓存候选记录，输出与候选名称无关的客户仓库归属签名。"""
    return tuple(map(int, np.asarray(record["candidate_customer_group"]).tolist()))


def _label_quality(record: dict[str, Any]) -> tuple[int, int, int]:
    """输入候选记录，输出用于精确标签优先替换的质量元组。"""
    target_mask = np.asarray(record["target_mask"], dtype=bool)
    group_mask = np.asarray(record["group_time_mask"], dtype=bool)
    return (
        int(not bool(record["right_censored"])),
        int(target_mask.sum()),
        int(group_mask.sum()),
    )


def merge_partition_gnn_caches(
    base_cache: dict[str, Any],
    addition_caches: Iterable[dict[str, Any]],
    *,
    source_names: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    按实例和完整客户归属合并多个GNN缓存。

    输入基础缓存、新增缓存及来源名称；输出新缓存和加入/替换/跳过统计。
    相同划分不会因候选名称不同而重复；新标签质量更高时替换旧标签。
    """
    additions = list(addition_caches)
    sources = tuple(map(str, source_names))
    base_features = base_cache["feature_names"]
    base_k = int(base_cache["k_neighbors"])
    records = list(base_cache["records"])
    instances = dict(base_cache["instances"])

    # 索引值是records中的位置，便于发现更高质量标签时原位替换。
    signature_index: dict[tuple[str, tuple[int, ...]], int] = {
        (str(record["instance_id"]), _partition_signature(record)): index
        for index, record in enumerate(records)
    }
    added = replaced = skipped = 0
    for cache in additions:
        if cache["feature_names"] != base_features:
            raise ValueError("新增GNN缓存的特征模式与基础缓存不一致。")
        if int(cache["k_neighbors"]) != base_k:
            raise ValueError("新增GNN缓存的道路近邻数与基础缓存不一致。")
        for instance_id, static in cache["instances"].items():
            instances.setdefault(str(instance_id), static)
        for record in cache["records"]:
            key = (str(record["instance_id"]), _partition_signature(record))
            old_index = signature_index.get(key)
            if old_index is None:
                signature_index[key] = len(records)
                records.append(record)
                added += 1
            elif _label_quality(record) > _label_quality(records[old_index]):
                records[old_index] = record
                replaced += 1
            else:
                skipped += 1

    previous_sources = tuple(map(str, base_cache.get("augmentation_sources", ())))
    merged = {
        **base_cache,
        "instances": instances,
        "records": records,
        "augmentation_sources": list(dict.fromkeys(previous_sources + sources)),
    }
    report = {
        "base_record_count": len(base_cache["records"]),
        "addition_cache_count": len(additions),
        "added_record_count": added,
        "replaced_record_count": replaced,
        "skipped_duplicate_count": skipped,
        "merged_record_count": len(records),
        "merged_instance_count": len(instances),
        "sources": list(sources),
    }
    return merged, report
