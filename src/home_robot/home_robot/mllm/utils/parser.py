import ast
import re
from typing import Any, Dict, List

import json_repair

from .filter import exclude_objects_by_category


class ParseError(RuntimeError):
    """后处理阶段专用异常，方便上游捕获"""

    pass


def default_json_parser(raw_ansewer: str) -> Dict[str, Any]:
    json_answer = json_repair.loads(raw_ansewer)
    return json_answer


def _extract_dict(raw: str) -> Dict[str, Any]:
    """从模型返回中提取最外层字典并 literal_eval"""
    if not raw:
        raise ParseError("模型返回为空")

    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        raise ParseError("未能定位到字典结构")

    try:
        data = ast.literal_eval(m.group(0))
    except Exception as e:
        raise ParseError(f"ast.literal_eval 失败: {e}")

    if not isinstance(data, dict):
        raise ParseError("顶层结构不是 dict")
    return data


def _check_field(data: Dict[str, Any], key: str, expected_type: type) -> Any:
    """通用字段类型检查"""
    val = data.get(key)
    if val is None:
        raise ParseError(f'缺少字段 "{key}"')
    if not isinstance(val, expected_type):
        raise ParseError(
            f'字段 "{key}" 应为 {expected_type.__name__}，实际类型 {type(val)}'
        )
    return val


def parse_merge_instance_output(raw: str) -> Dict[str, Any]:
    """
    解析合并实例的输出，提取并验证merged_type和merged_description字段
    {{
    "merged_type": "merged type",
    "merged_description": "merged description"
    }}
    """
    if not raw:
        raise ParseError("模型返回为空")

    # 提取最外层字典
    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        raise ParseError("未能定位到字典结构")
    dict_str = m.group(0)

    # 安全解析为Python字典
    try:
        data = ast.literal_eval(dict_str)
    except Exception as e:
        raise ParseError(f"ast.literal_eval 失败: {e}")

    if not isinstance(data, dict):
        raise ParseError("顶层结构不是 dict")

    # 字段校验与类型转换
    def _check_str(key: str) -> str:
        val = data.get(key)
        if val is None:
            raise ParseError(f'缺少字段 "{key}"')
        if not isinstance(val, str):
            raise ParseError(f'字段 "{key}" 应为 str，实际类型 {type(val)}')
        return val

    cleaned = {
        "merged_category": _check_str("merged_category"),
        "merged_description": _check_str("merged_description"),
    }

    return cleaned


def parse_instance_similarity_output(raw: str) -> Dict[str, Any]:
    try:
        data = ast.literal_eval(raw)
    except Exception as e:
        raise ParseError(f"ast.literal_eval 失败: {e}")

    if not isinstance(data, dict):
        raise ParseError("顶层结构不是 dict")

    # 字段校验与类型转换
    def _check_str(key: str) -> str:
        val = data.get(key)
        if not isinstance(val, str):
            raise ParseError(f'字段 "{key}" 应为 str，实际类型 {type(val)}')
        return val

    def _check_bool(key: str) -> bool:
        val = data.get(key)
        if not isinstance(val, bool):
            raise ParseError(f'字段 "{key}" 应为 bool，实际类型 {type(val)}')
        return val

    cleaned = {
        "reasoning": _check_str("reasoning"),
        "should_merge": _check_bool("should_merge"),
    }

    return cleaned


def parse_grounding_output(raw: str) -> Dict[str, Any]:
    """
    解析 FOR_GROUNDING_ZH 提示下模型返回的字符串，返回结构化 dict。
    顶层字段参考 prompts 中的 FOR_GROUNDING 提示
    如果解析失败，抛出 ParseError。
    """
    cleaned = json_repair.loads(raw)
    cleaned = exclude_objects_by_category(cleaned=cleaned)
    return cleaned


def parse_checking_output(raw: str) -> Dict[str, Any]:
    """
    解析 FOR_CHECKING 提示下模型返回的字符串，返回结构化 dict。
    顶层字段参考 prompts 中的 FOR_CHECKING 提示
    如果解析失败，抛出 ParseError。
    """
    if not raw:
        raise ParseError("模型返回为空")

    # 提取最外层字典
    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        raise ParseError("未能定位到字典结构")
    dict_str = m.group(0)

    # 安全解析
    try:
        data = ast.literal_eval(dict_str)
    except Exception as e:
        raise ParseError(f"ast.literal_eval 失败: {e}")

    if not isinstance(data, dict):
        raise ParseError("顶层结构不是 dict")

    # 字段校验
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str):
        raise ParseError(f'"reasoning" 应为 str，实际类型 {type(reasoning)}')

    answer = data.get("answer")
    if not isinstance(answer, bool):
        raise ParseError(f'"answer" 应为 bool，实际类型 {type(answer)}')

    cleaned = {
        "reasoning": reasoning,
        "answer": answer,
    }
    return cleaned


def get_objects_type_list(objects: List[Dict[str, Any]]) -> List[str]:
    if not objects:
        raise ParseError("输入")
    types = [obj["type"] for obj in objects]
    return types


def parse_renew_instance_output(raw: str) -> Dict[str, Any]:
    """
    解析 renew_instance 的输出
    返回: {"category": str, "description": str}
    """
    data = _extract_dict(raw)
    return {
        "category": _check_field(data, "category", str),
        "description": _check_field(data, "description", str),
    }


def parse_group_description_output(raw: str) -> Dict[str, Any]:
    """
    解析 generate_group_description 的输出
    返回: {"group_description": str}
    """
    data = _extract_dict(raw)
    return {
        "group_description": _check_field(data, "group_description", str),
    }


def parse_value_output(raw: str) -> Dict[str, Any]:
    """
    解析 generate_group_description 的输出
    返回: {"group_description": str}
    """
    data = _extract_dict(raw)
    return {
        "answer": _check_field(data, "answer", int),
    }
