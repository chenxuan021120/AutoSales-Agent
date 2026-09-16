"""规划器：目前只允许调用本地知识库，结构已为未来扩展网页搜索预留。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import settings


@dataclass
class Action:
    tool: str
    query: str
    reason: str


def _rule_plan(user_query: str) -> list[Action]:
    local_terms = ("es9", "星辰", "续航", "充电", "座舱", "驾驶", "配置", "电池", "售后")
    web_terms = ("最新", "最近", "新闻", "市场", "价格", "销量", "竞品", "比较", "华为", "问界", "小米", "特斯拉", "小鹏", "理想", "蔚来")
    query_lower = user_query.lower()
    if any(word in user_query for word in ("先别算", "先不算", "暂不计算", "暂时不算", "不要计算", "先不要计算")) and not any(word in user_query for word in ("银行", "工行", "建行", "金融方案", "官方方案", "分期方案", "ES9", "星辰", "充电", "续航", "座舱", "配置")):
        return []
    # 明确的闲聊、确认和结束语不应因为默认分支而触发 RAG。
    no_tool_markers = ("好看吗", "帅不帅", "谢谢", "感谢", "好的", "知道了", "明白了", "先不用查", "先别查", "不用查", "不用了", "暂时不看", "提车祝福", "心情不错", "一定会", "能保证", "不会后悔")
    factual_markers = ("参数", "配置", "多少", "多久", "是否", "有没有", "支持", "续航", "充电", "价格", "风阻", "质保", "怎么", "查询", "比较", "贷款", "月供", "首付", "金融", "银行", "分期", "利率")
    if any(marker in user_query for marker in no_tool_markers) and not any(marker in user_query for marker in factual_markers):
        return []
    if "想贷款" in user_query and not any(marker in user_query for marker in ("计算", "算一下", "月供", "方案", "利率")):
        return []
    if any(phrase in user_query for phrase in ("没有找到银行官网链接", "找不到银行官网链接", "没有银行官网链接", "没有官方链接")):
        return []
    if user_query.startswith("我在") and "想贷款" in user_query and not any(marker in user_query for marker in ("计算", "算一下", "月供", "方案", "利率")):
        return []
    if "不看竞品" in user_query and not any(marker in user_query for marker in ("比较", "价格", "参数", "区别")):
        return []
    if user_query.strip() in {"好的", "好", "谢谢", "感谢", "知道了", "明白了", "行"}:
        return []
    if any(prefix in user_query for prefix in ("我在", "我的预算", "预算", "我考虑", "我想买", "改成", "调整为")) and not any(marker in user_query for marker in factual_markers):
        return []
    if "最重要的是什么" in user_query and not any(marker in user_query for marker in ("ES9", "参数", "配置")):
        return []
    use_local = any(term in query_lower for term in local_terms)
    use_web = any(term in query_lower for term in web_terms)
    if any(term in user_query for term in ("不要联网", "不要查网页", "只看手册", "只看资料")):
        use_web = False
    if any(term in user_query for term in ("不需要竞品", "不用查竞品", "只看ES9")):
        use_web = False
    bank_terms = ("工行", "工商银行", "建行", "建设银行", "中行", "中国银行", "农行", "农业银行", "交行", "交通银行", "招行", "招商银行", "银行官网")
    finance_terms = ("银行贷款", "银行方案", "官方方案", "金融方案", "分期方案", "分期利率", "贷款利率", "年化利率", "费率")
    use_finance = (
        any(term in user_query for term in finance_terms)
        or ("车贷" in user_query and any(term in user_query for term in ("查询", "联网", "银行", "方案")))
        or (any(term in user_query for term in bank_terms) and any(term in user_query for term in ("分期", "贷款", "首付", "利率", "费率", "比较", "不计算", "月供")))
    )
    # 金融问题默认只走金融官方检索；只有同一问题明确含竞品/车型实时信息时才保留普通 Web。
    competitor_terms = ("问界", "M9", "小鹏", "理想", "蔚来", "特斯拉", "竞品", "车型")
    explicit_nonfinance = any(term in user_query for term in ("销量", "新闻", "市场", "竞品", "最近价格", "最新价格", "对比"))
    if use_finance and not any(term in user_query for term in competitor_terms) and not explicit_nonfinance:
        use_web = False
    paused_calculation = any(term in user_query for term in ("先别算", "先不算", "暂不计算", "暂时不算", "不要计算", "先不要计算"))
    use_calculator = (not paused_calculation) and any(term in user_query for term in ("贷款", "月供", "首付", "电费", "电耗", "能耗", "预算", "落地价")) and (not use_finance or any(term in user_query for term in ("计算", "算一下", "月供")))

    actions: list[Action] = []
    if use_calculator:
        actions.append(Action("car_calculator", user_query, "问题涉及贷款、能耗或购车预算，调用确定性计算工具"))
    if (use_local and (not use_finance or any(term in user_query for term in competitor_terms))) or (not use_web and not use_calculator and not use_finance):
        actions.append(Action("local_rag", user_query, "问题涉及产品资料，查询本地知识库"))
    if use_web:
        actions.append(Action("web_search", user_query, "问题涉及实时信息或竞品，查询网页"))
    if use_finance:
        actions.append(Action("finance_web_search", user_query, "问题涉及实时银行或品牌金融方案，只查询官方页面"))
    return actions


def plan(user_query: str) -> list[Action]:
    """让 LLM 选择工具；网络异常或格式异常时回退到透明规则。"""
    rule_actions = _rule_plan(user_query)
    # 银行之间的方案比较仍然只走官方金融检索，避免“比较”额外触发泛网页搜索。
    if any(action.tool == "finance_web_search" for action in rule_actions) and not any(action.tool == "car_calculator" for action in rule_actions) and any(term in user_query for term in ("工行", "工商银行", "建行", "建设银行", "中行", "农行", "交行", "招行")) and not any(action.tool in {"local_rag", "web_search"} for action in rule_actions):
        return [action for action in rule_actions if action.tool == "finance_web_search"]
    # ponytail: 明确的银行方案查询无需模型二次判断，避免“首付”被误路由到计算器。
    if len(rule_actions) == 1 and rule_actions[0].tool == "finance_web_search":
        return rule_actions
    if not settings.openai_api_key:
        return rule_actions

    prompt = f'''你是汽车销售 Agent 的规划器。只输出 JSON，不要 Markdown。
用户问题：{user_query}
可用工具：
- local_rag：查询星辰电动 ES9 的本地产品资料。
- web_search：查询竞品、最新价格、市场新闻或本地资料没有的信息。
- car_calculator：计算贷款月供、年度电费或购车预算；只能用于数学测算，不能猜测缺失参数。
- finance_web_search：查询实时汽车金融方案；只检索银行官网，不推荐机构、不猜测最终利率。
规则：ES9 产品参数优先 local_rag；涉及竞品或实时信息使用 web_search；跨车型比较通常同时使用两者；涉及贷款、能耗、预算时使用 car_calculator；涉及银行或品牌金融方案时使用 finance_web_search。
输出格式：{{"actions":[{{"tool":"local_rag、web_search、car_calculator 或 finance_web_search","query":"检索问题","reason":"简短原因"}}]}}'''
    endpoint = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(
            {
                "model": settings.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    print("[Plan] 正在请求 qwen-plus 生成工具计划…", flush=True)
    try:
        with urlopen(request, timeout=30) as response:
            content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
        payload = json.loads(content)
        actions = [
            Action(item["tool"], item.get("query") or user_query, item.get("reason") or "模型规划")
            for item in payload.get("actions", [])
            if item.get("tool") in {"local_rag", "web_search", "car_calculator", "finance_web_search"}
        ]
        # ponytail: 数学问题不能依赖模型是否“想起”调用工具，规则补上唯一的确定性工具。
        if any(action.tool == "car_calculator" for action in rule_actions) and not any(action.tool == "car_calculator" for action in actions):
            actions.extend(action for action in rule_actions if action.tool == "car_calculator")
        if any(action.tool == "finance_web_search" for action in rule_actions) and not any(action.tool == "finance_web_search" for action in actions):
            actions.extend(action for action in rule_actions if action.tool == "finance_web_search")
        return actions or rule_actions
    except (HTTPError, URLError, TimeoutError, OSError, KeyError, TypeError, ValueError):
        print("[Plan] 模型规划不可用，已回退到规则规划。", flush=True)
        return _rule_plan(user_query)
