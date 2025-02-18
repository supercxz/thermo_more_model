# 函数：解析化学成分及其比例
import re
from collections import defaultdict


def parse_original_formulas(formula):
    """
    将formula解析为化学成分及其比例。
    返回一个字典，键为化学成分，值为比例。
    """
    element_count = defaultdict(float)
    elements = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula)
    for element, ratio in elements:
        # 如果没有数量则默认为1
        count = float(ratio) if ratio else 1.0
        element_count[element] += count
    return dict(element_count)


def normalized_formulas(formula):
    """
    将formula归一化
    返回一个字符串
    """
    element_count = defaultdict(float)
    elements = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula)
    for element, ratio in elements:
        # 如果没有数量则默认为1
        count = float(ratio) if ratio else 1.0
        element_count[element] += count
    return dict(element_count)


def normalize_chemical_formula(formula):
    """
    将formula归一化
    Normalize a chemical formula to a format where the amount of each element
    is represented as a fraction of the whole, rounded to four decimal places, 
    and sums to 1. Handles both integers and decimal numbers in the formula.

    :param formula: A string representing the chemical formula (e.g., H2O or H2.5O1.5)
    :return: A normalized string representing the formula in fractional format
    """
    # Regular expression to find elements and their counts (including decimals)
    pattern = r'([A-Z][a-z]*)(\d+(\.\d+)?)?'  # Matches elements with optional decimals
    elements = defaultdict(float)

    # Parse the formula
    for match in re.findall(pattern, formula):
        element = match[0]  # 取第一个捕获组，即元素
        count = match[1]    # 取第二个捕获组，即数量（如果存在的话）
        elements[element] += float(count) if count else 1.0

    # Calculate total amount
    total = sum(elements.values())

    # Normalize and format the result
    normalized_parts = []
    for element, count in elements.items():
        ratio = count / total
        normalized_parts.append(f"{element}{ratio:.4f}")

    # Join the normalized parts into a single string
    result = ''.join(normalized_parts)
    return result


if __name__ == '__main__':
    print("测试：输入H2O")
    print(normalize_chemical_formula("H2O"))

    print("测试：输入H4.5O0.5Ti5")
    print(normalize_chemical_formula("H4.5O0.5Ti5"))

    print("测试：输入H")
    print(normalize_chemical_formula("H"))
