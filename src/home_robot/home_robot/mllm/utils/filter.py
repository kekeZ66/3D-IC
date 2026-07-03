import logging
import re
from typing import Any, Dict, List, Pattern

logger = logging.getLogger(__name__)

# 改成正则模式列表，支持更灵活的匹配
EXCLUDE_PATTERNS = [
    r"^door$",  # 精确匹配 "door"
    r"door\s*frame",  # 匹配 "door frame" 或 "doorframe"
    r"^ceiling",  # 匹配所有以 "ceiling" 开头的类别
    r"wall",  # 包含 "wall" 的词
    r"window",  # 包含 "window" 的词
    r"outlet",  # 包含 "outlet" 的词
    r"baseboard",  # 包含 "baseboard" 的词
    r"carpet",  # 包含 "carpet" 的词
    r"curtain",  # 包含 "curtain" 的词
    r"background",  # 包含 "background" 的词
    r"mirror",  # 包含 "mirror" 的词
    r"doorway",  # 包含 "doorway" 的词
    r"headboard",  # 包含 "headboard" 的词
]


def _compile_patterns(patterns: List[str]) -> List[Pattern]:
    """预编译正则表达式以提高性能"""
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


COMPILED_PATTERNS = _compile_patterns(EXCLUDE_PATTERNS)


def _is_excluded(category: str, patterns: List[Pattern]) -> bool:
    """检查类别是否匹配任何一个排除模式"""
    return any(pattern.search(category) for pattern in patterns)


def exclude_objects_by_category(
    cleaned: Dict[str, Any], exclude_patterns: List[Pattern] = COMPILED_PATTERNS
) -> Dict[str, Any]:
    """
    从 cleaned 数据中过滤掉匹配指定正则模式的物体类别
    """
    # 复制原始数据以避免修改输入
    filtered = {**cleaned}

    # 过滤objects列表，保留不匹配任何排除模式的物体
    filtered_objects = [
        obj
        for obj in cleaned["objects"]
        if not _is_excluded(obj.get("category", ""), exclude_patterns)
    ]
    filtered["objects"] = filtered_objects
    logger.debug(f"过滤掉了 {len(cleaned['objects']) - len(filtered_objects)} 个物体")

    # 获取保留下来的物体ID
    remaining_ids = {obj["instance_id"] for obj in filtered_objects}

    # 过滤groups中的instances，只保留存在的物体ID
    filtered_regions = []
    for region in cleaned["regions"]:
        filtered_instances = [
            instance_id
            for instance_id in region["instances"]
            if instance_id in remaining_ids
        ]
        # 只保留包含至少两个物体的组（符合原始要求）
        if len(filtered_instances) >= 2:
            filtered_region = {**region}
            filtered_region["instances"] = filtered_instances
            filtered_regions.append(filtered_region)
    filtered["regions"] = filtered_regions

    return filtered
