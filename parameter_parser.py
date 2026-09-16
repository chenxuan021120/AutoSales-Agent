"""把购车口语参数标准化为确定性计算器可用的 schema。

该模块只做解析，不执行计算，也不替用户补齐缺失参数。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from decimal import Decimal


CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def chinese_number(value: str) -> Decimal | None:
    value = value.strip()
    for suffix, multiplier in (("千", 1000), ("百", 100), ("万", 10000)):
        if value.endswith(suffix):
            prefix = value[:-1]
            parsed = chinese_number(prefix) if prefix else Decimal(1)
            if parsed is not None:
                return parsed * multiplier
    if value.endswith("成") and value[:-1] in CN_DIGITS:
        return Decimal(CN_DIGITS[value[:-1]]) / 10
    if value in CN_DIGITS:
        return Decimal(CN_DIGITS[value])
    if value == "十":
        return Decimal(10)
    if "十" in value:
        left, right = value.split("十", 1)
        return Decimal((CN_DIGITS.get(left, 1) * 10) + CN_DIGITS.get(right, 0))
    return None


def number(value: str) -> Decimal:
    return chinese_number(value) or Decimal(value)


@dataclass
class NormalizedParameters:
    car_price: float | None = None
    down_payment_ratio: float | None = None
    loan_term_months: int | None = None
    annual_interest_rate: float | None = None
    annual_mileage_km: float | None = None
    consumption_kwh_per_100km: float | None = None
    electricity_price_per_kwh: float | None = None
    insurance: float | None = None
    registration_fee: float | None = None
    purchase_tax: float | None = None
    options: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _money(query: str, label: str) -> Decimal | None:
    match = re.search(rf"{label}(?:费|金额)?\s*(\d+(?:\.\d+)?|[零〇一二两三四五六七八九十]+)\s*(万|千|百|元)?", query)
    if not match:
        return None
    value = number(match.group(1))
    multiplier = {"万": Decimal("10000"), "千": Decimal("1000"), "百": Decimal("100"), "元": Decimal("1"), None: Decimal("1")}[match.group(2)]
    return value * multiplier


def parse_parameters(query: str) -> NormalizedParameters:
    """解析可识别字段；未出现的字段保持 None。"""
    result = NormalizedParameters()

    price = re.search(r"(?:车价|裸车价|预算)\s*(\d+(?:\.\d+)?)\s*万", query)
    if price:
        result.car_price = float(Decimal(price.group(1)) * 10000)

    down = re.search(r"(?:首付(?:比例)?\s*)(\d+(?:\.\d+)?)\s*%|\b(\d+(?:\.\d+)?)\s*%\s*首付|(?:首付\s*)?([零〇一二两三四五六七八九十])成", query)
    if down:
        raw = down.group(1) or down.group(2) or down.group(3)
        result.down_payment_ratio = float(Decimal(raw) / 100 if down.group(1) or down.group(2) else number(raw) / 10)

    loan_context = any(term in query for term in ("贷款", "贷", "分期", "月供", "首付", "利率", "个月", "按"))
    if loan_context:
        years = re.search(r"(?:贷款|贷|分期)?\s*([0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十]+)\s*年", query)
        months = re.search(r"([0-9]+)\s*个月", query)
        if months:
            result.loan_term_months = int(months.group(1))
        elif years:
            result.loan_term_months = int(number(years.group(1)) * 12)

        rate = re.search(r"(?:年利率|利率|按|按照)\s*([0-9]+(?:\.[0-9]+)?)\s*%?", query)
        if rate:
            # 口语中的 3.5 / 3.5% 都是 3.5 个百分点，统一转成 0.035。
            result.annual_interest_rate = float(Decimal(rate.group(1)) / 100)

    mileage = re.search(r"(?:每年|一年|年均)\s*([0-9]+(?:\.[0-9]+)?)\s*(万)?\s*(?:公里|km)", query, re.I)
    if mileage:
        result.annual_mileage_km = float(Decimal(mileage.group(1)) * (10000 if mileage.group(2) else 1))

    # “总共用了18度电”不是百公里电耗。
    if not re.search(r"(?:总共|一共|用了)\s*[0-9.]+\s*(?:度|kwh)", query, re.I):
        energy = re.search(r"(?:百公里|电耗\s*)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:度|kwh)\s*(?:/\s*100\s*(?:公里|km))?", query, re.I)
        if energy and not ("百公里" in query or "电耗" in query or "/100" in query.lower() or "/ 100" in query.lower()):
            energy = None
        if energy:
            result.consumption_kwh_per_100km = float(energy.group(1))

    electricity = re.search(r"(?:电价\s*)?([0-9]+(?:\.[0-9]+)?)\s*元\s*(?:一度|每度|/\s*(?:度|kwh))|电价\s*([0-9]+(?:\.[0-9]+)?)\s*元", query, re.I)
    if electricity:
        result.electricity_price_per_kwh = float(electricity.group(1) or electricity.group(2))

    for field, label in (("insurance", "保险"), ("registration_fee", "上牌"), ("purchase_tax", "购置税"), ("options", "选装")):
        value = _money(query, label)
        if value is not None:
            setattr(result, field, float(value))
    return result
