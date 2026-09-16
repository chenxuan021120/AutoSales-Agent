"""购车测算工具：金额计算只由 Python 完成，不交给大模型。"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from parameter_parser import parse_parameters


class CalculationError(ValueError):
    """用户给出的计算参数不完整或不合理。"""


def missing_calculation_parameters(query: str, loan_slots: dict[str, str] | None = None) -> list[str]:
    """计算器执行前的轻量 preflight；只报告缺参，不执行计算。"""
    normalized = parse_parameters(query)
    if any(word in query for word in ("贷款", "月供", "首付")):
        slots = loan_slots or {}
        missing = []
        if not normalized.car_price and "car_price" not in slots and not re.search(r"(?:车价|裸车价|价格)?\s*\d+(?:\.\d+)?\s*万", query):
            missing.append("car_price")
        if "down_payment" not in slots and normalized.down_payment_ratio is None:
            missing.append("down_payment")
        if "years" not in slots and normalized.loan_term_months is None:
            missing.append("years")
        if "annual_rate" not in slots and normalized.annual_interest_rate is None:
            missing.append("annual_rate")
        return missing
    if any(word in query for word in ("电费", "电耗", "能耗", "电价")):
        missing = []
        if normalized.annual_mileage_km is None:
            missing.append("annual_mileage_km")
        if normalized.consumption_kwh_per_100km is None:
            missing.append("consumption_kwh_per_100km")
        if normalized.electricity_price_per_kwh is None:
            missing.append("electricity_price_per_kwh")
        return missing
    if any(word in query for word in ("预算", "落地价")):
        missing = []
        if normalized.car_price is None:
            missing.append("car_price")
        if "购置税" in query and normalized.purchase_tax is None:
            missing.append("purchase_tax")
        return missing
    return []


CALCULATION_PROMPTS = {
    "car_price": "车价（例如 30 万）",
    "down_payment": "首付比例（例如 首付 30%）",
    "years": "贷款年限（例如 3 年）",
    "annual_rate": "年利率（例如 3.5%）",
    "annual_mileage_km": "年行驶里程（例如每年 1.5 万公里）",
    "consumption_kwh_per_100km": "百公里电耗（例如 18 度/100公里）",
    "electricity_price_per_kwh": "电价（例如 0.6 元/度）",
    "purchase_tax": "购置税金额（例如 购置税 0 元）",
}


def missing_parameter_prompt(names: list[str]) -> str:
    return "计算工具需要补充参数：缺少" + "、".join(CALCULATION_PROMPTS[name] for name in names) + "。"


_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_number(value: str) -> Decimal | None:
    """解析常见中文整数/成数，不试图覆盖复杂中文财务数字。"""
    value = value.strip()
    if value == "半":
        return Decimal("0.5")
    if value.endswith("成") and value[:-1] in _CN_DIGITS:
        return Decimal(_CN_DIGITS[value[:-1]]) / 10
    if value in _CN_DIGITS:
        return Decimal(_CN_DIGITS[value])
    if value == "十":
        return Decimal(10)
    if "十" in value:
        left, right = value.split("十", 1)
        return Decimal((_CN_DIGITS.get(left, 1) * 10) + _CN_DIGITS.get(right, 0))
    return None


def _number_or_cn(value: str) -> Decimal:
    parsed = _cn_number(value)
    return parsed if parsed is not None else Decimal(value)


def _money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,.2f} 元"


def _positive(name: str, value: Decimal) -> None:
    if value <= 0:
        raise CalculationError(f"{name}必须大于 0。")


def calculate_loan(car_price: float, down_payment_ratio: float, annual_rate: float, years: float) -> str:
    """等额本息月供；比例参数使用小数，如 30% 传 0.30。"""
    price, down, rate, term = map(Decimal, map(str, (car_price, down_payment_ratio, annual_rate, years)))
    _positive("车价", price)
    _positive("贷款年限", term)
    if not Decimal("0") <= down < Decimal("1"):
        raise CalculationError("首付比例应在 0% 到 100% 之间。")
    if rate < 0:
        raise CalculationError("年利率不能为负数。")
    principal = price * (Decimal("1") - down)
    months = int(term * 12)
    monthly_rate = rate / 12
    monthly_payment = principal / months if monthly_rate == 0 else principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    total_payment = monthly_payment * months
    return "\n".join((
        "贷款测算（等额本息）：",
        f"- 贷款本金：{_money(principal)}",
        f"- 月供：{_money(monthly_payment)} × {months} 期",
        f"- 总利息：{_money(total_payment - principal)}",
        f"- 本息合计：{_money(total_payment)}",
        "注：这是示例测算，不等同于银行或金融机构的最终报价。",
    ))


def calculate_energy_cost(annual_mileage: float, energy_per_100km: float, electricity_price: float) -> str:
    mileage, energy, price = map(Decimal, map(str, (annual_mileage, energy_per_100km, electricity_price)))
    for name, value in (("年行驶里程", mileage), ("百公里电耗", energy), ("电价", price)):
        _positive(name, value)
    annual_kwh = mileage / 100 * energy
    return "\n".join((
        "年度电费测算：",
        f"- 年用电量：{annual_kwh.quantize(Decimal('0.01')):,.2f} 度",
        f"- 年电费：{_money(annual_kwh * price)}",
        "注：未包含停车费、服务费；实际电耗会受温度、路况和驾驶习惯影响。",
    ))


def calculate_budget(car_price: float, insurance: float = 0, purchase_tax: float = 0, registration: float = 0, options: float = 0) -> str:
    values = [Decimal(str(value)) for value in (car_price, insurance, purchase_tax, registration, options)]
    if any(value < 0 for value in values):
        raise CalculationError("预算项目不能为负数。")
    _positive("车价", values[0])
    total = sum(values)
    return "\n".join((
        "购车预算测算：",
        f"- 车辆价格：{_money(values[0])}",
        f"- 保险：{_money(values[1])}；购置税：{_money(values[2])}；上牌：{_money(values[3])}；选装：{_money(values[4])}",
        f"- 预计落地总价：{_money(total)}",
        "注：税费、保险与地方政策会变化，请以实际报价为准。",
    ))


def _number(pattern: str, query: str, name: str, unit: Decimal = Decimal("1")) -> Decimal:
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        raise CalculationError(f"缺少{name}。")
    return Decimal(match.group(1)) * unit


def _loan_value(pattern: str, query: str, name: str, unit: Decimal = Decimal("1"), cn: bool = True) -> Decimal:
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        raise CalculationError(f"缺少{name}。")
    return _number_or_cn(match.group(1)) * unit if cn else Decimal(match.group(1)) * unit


def _amount_after(label: str, query: str, required: bool = False) -> Decimal:
    match = re.search(rf"{label}(?:费|金额)?\s*(\d+(?:\.\d+)?|[零〇一二两三四五六七八九十]+)\s*(万|元)?", query)
    if not match:
        if required:
            raise CalculationError(f"缺少{label}金额（例如 {label} 6000 元）。")
        return Decimal("0")
    amount = _number_or_cn(match.group(1))
    return amount * (Decimal("10000") if match.group(2) == "万" else Decimal("1"))


def calculate_from_query(query: str) -> str:
    """识别三种教学用自然语言输入；参数不足时给出明确追问。"""
    try:
        normalized = parse_parameters(query)
        if any(word in query for word in ("贷款", "月供", "首付")):
            if "贷款本金" in query and not re.search(r"(?:车价|裸车价)\s*\d+(?:\.\d+)?\s*万", query):
                raise CalculationError("缺少车价（例如 30 万）")
            car_price = Decimal(str(normalized.car_price)) if normalized.car_price is not None else _loan_value(r"(?:车价|裸车价|预算)?\s*(\d+(?:\.\d+)?)\s*万", query, "车价（例如 30 万）", Decimal("10000"), cn=False)
            pct_match = re.search(r"首付\s*(\d+(?:\.\d+)?)\s*%", query)
            down = Decimal(pct_match.group(1)) * Decimal("0.01") if pct_match else Decimal("0")
            cn_down = re.search(r"(?:首付\s*|^)([零〇一二两三四五六七八九十]+)成", query)
            if cn_down:
                down = _number_or_cn(cn_down.group(1)) / 10
            if not pct_match and not cn_down:
                raise CalculationError("首付比例（例如 首付 30%）")
            years_match = re.search(r"(?:贷款|贷|分期)?\s*(\d+(?:\.\d+)?)\s*年", query)
            cn_years = re.search(r"(?:贷款|贷|分期)?\s*([零〇一二两三四五六七八九十]+)年", query)
            years = Decimal(normalized.loan_term_months) / 12 if normalized.loan_term_months is not None else (Decimal(years_match.group(1)) if years_match else (_number_or_cn(cn_years.group(1)) if cn_years else Decimal("0")))
            if not years:
                months = _loan_value(r"(\d+(?:\.\d+)?)\s*个月", query, "贷款年限（例如 36 个月）", cn=False)
                years = months / 12
            if not years:
                raise CalculationError("贷款年限（例如 3 年）")
            rate_match = re.search(r"(?:年利率|利率)\s*(\d+(?:\.\d+)?)\s*%?", query)
            if not rate_match:
                raise CalculationError("年利率（例如 年利率 3.5%）")
            rate = Decimal(str(normalized.annual_interest_rate)) if normalized.annual_interest_rate is not None else Decimal(rate_match.group(1)) * Decimal("0.01")
            return calculate_loan(float(car_price), float(down), float(rate), float(years))
        if any(word in query for word in ("电费", "电耗", "能耗", "电价")):
            mileage_match = re.search(r"(?:每年|一年|年均)?\s*(\d+(?:\.\d+)?)\s*(万)?\s*(?:公里|km)", query, re.IGNORECASE)
            if not mileage_match:
                raise CalculationError("缺少年行驶里程（例如每年 1.5 万公里）。")
            mileage = Decimal(str(normalized.annual_mileage_km)) if normalized.annual_mileage_km is not None else Decimal(mileage_match.group(1))
            if normalized.annual_mileage_km is None and mileage_match.group(2):
                mileage *= Decimal("10000")
            if re.search(r"(?:总共|一共|用了)\s*\d+(?:\.\d+)?\s*(?:度|kwh)", query, re.IGNORECASE):
                raise CalculationError("缺少百公里电耗（例如 18 度/100公里）。")
            energy_match = re.search(r"(?:电耗\s*)?(\d+(?:\.\d+)?)\s*(?:度|kwh)(?:\s*/?\s*100\s*(?:公里|km))?", query, re.IGNORECASE)
            if not energy_match or not ("电耗" in query or "/100" in query or "百公里" in query):
                raise CalculationError("缺少百公里电耗（例如 18 度/100公里）。")
            energy = Decimal(str(normalized.consumption_kwh_per_100km)) if normalized.consumption_kwh_per_100km is not None else Decimal(energy_match.group(1))
            price_match = re.search(r"(?:电价\s*)?(\d+(?:\.\d+)?)\s*元\s*(?:/\s*(?:度|kwh)|一度|每度)?", query, re.IGNORECASE)
            if not price_match or not any(word in query for word in ("电价", "每度", "一度", "元/kwh", "元/度")):
                raise CalculationError("缺少电价（例如 0.6 元/度）。")
            price = Decimal(str(normalized.electricity_price_per_kwh)) if normalized.electricity_price_per_kwh is not None else Decimal(price_match.group(1))
            return calculate_energy_cost(float(mileage), float(energy), float(price))
        if any(word in query for word in ("预算", "落地价")):
            if "购置税" in query and not re.search(r"购置税\s*\d+(?:\.\d+)?\s*(?:万|元)", query):
                raise CalculationError("缺少购置税金额（例如 购置税 0 元）。")
            return calculate_budget(
                float(_amount_after("车价", query) or _amount_after("预算", query, required=True)),
                float(_amount_after("保险", query)),
                float(_amount_after("购置税", query)),
                float(_amount_after("上牌", query)),
                float(_amount_after("选装", query)),
            )
        raise CalculationError("请说明要测算贷款、年度电费，还是购车预算。")
    except CalculationError as error:
        return f"计算工具需要补充参数：{error}"
