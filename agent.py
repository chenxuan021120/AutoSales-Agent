"""Agent 编排层：plan -> tool -> memory -> answer。"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from answer import answer
from calculators import calculate_from_query, missing_calculation_parameters, missing_parameter_prompt
from parameter_parser import parse_parameters
from config import settings
from embeddings import get_embedder
from graph import JsonGraph
from planner import plan
from web_search import finance_search, search as web_search
from eval.evidence_schema import make_evidence
from failure_injection import injected_failure
from memory_store import MemoryManager, MemoryStoreError


@dataclass
class ConversationState:
    """CLI 会话中的短期状态；退出程序即清除，不保存个人信息。"""

    pending_tool: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    profile: dict[str, str] = field(default_factory=dict)
    recent_turns: list[str] = field(default_factory=list)
    vision_memory: list[str] = field(default_factory=list)
    vision_summary: str | None = None
    session_id: str | None = None
    memory_manager: MemoryManager | None = None
    memory_loaded: bool = False
    memory_loaded_fields: list[str] = field(default_factory=list)
    memory_updated_fields: list[str] = field(default_factory=list)
    memory_error: str | None = None
    # P1.3 仅供测试传入；默认 None，生产与原有评测不会注入故障。
    failure_injection: dict | None = None
    # 最近一次请求的轻量执行记录，供行为评测读取；不改变现有 CLI 行为。
    last_trace: dict = field(default_factory=dict)


def load_session_state(session_id: str, memory_manager: MemoryManager | None = None) -> ConversationState:
    """从 SQLite 恢复一个明确 Session；不存在时返回空状态。"""
    manager = memory_manager or MemoryManager()
    state = ConversationState(session_id=session_id, memory_manager=manager)
    try:
        payload = manager.load_memory(session_id)
        state.profile.update(payload["profile"])
        state.recent_turns[:] = payload["recent_turns"][-10:]
        state.vision_summary = payload.get("vision_summary")
        state.memory_loaded = True
        state.memory_loaded_fields = sorted(payload["profile"])
        if state.vision_summary:
            state.memory_loaded_fields.append("vision_summary")
    except MemoryStoreError as error:
        state.memory_error = str(error)
    return state


def clear_session_memory(state: ConversationState) -> None:
    """仅删除当前 session 的持久化字段，运行时对象同步清空。"""
    if state.session_id and state.memory_manager:
        state.memory_manager.clear_memory(state.session_id)
    state.profile.clear()
    state.recent_turns.clear()
    state.vision_summary = None
    state.memory_updated_fields = []


def _update_profile(text: str, state: ConversationState) -> None:
    """只记录用户主动说出的少量购车偏好，不调用模型推断个人信息。"""
    patterns = {
        "city": r"(?:在|城市是|坐标在)\s*([\u4e00-\u9fff]{2,8})",
        "budget": r"(?:预算\s*(?:是|大概)?|改成|调整为)\s*(\d+(?:\.\d+)?)\s*万",
        "purchase_timeline": r"(今年|明年|本月|近期|三个月内|半年内|年底)\s*(?:买车|购车)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match and match.group(1) not in {"什么", "哪里", "多少"}:
            state.profile[name] = match.group(1) + ("万" if name == "budget" else "")
    if any(word in text for word in ("全款", "不贷款")):
        state.profile["loan_preference"] = "全款"
    elif "贷款" in text or "分期" in text:
        state.profile["loan_preference"] = "贷款/分期"
    model = re.search(r"(?:关注|考虑|想买|意向车型是)\s*([\u4e00-\u9fffA-Za-z0-9-]{2,12})", text)
    if model and not model.group(1).startswith(("什么", "哪", "多少")):
        state.profile["interested_model"] = model.group(1)
    approximate = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:万)?\s*左右[。！!]?", text)
    if approximate and any(word in " ".join(state.recent_turns) for word in ("银行", "方案", "金融")):
        state.profile["budget"] = approximate.group(1) + "万"


def _profile_memory(state: ConversationState) -> str:
    if not state.profile:
        return ""
    labels = {"city": "城市", "budget": "预算", "purchase_timeline": "购车时间", "interested_model": "关注车型", "loan_preference": "贷款偏好"}
    return "；".join(f"{labels[key]}：{value}" for key, value in state.profile.items() if key in labels)


def _memory_lookup_response(query: str, state: ConversationState) -> str | None:
    """只回答用户主动询问的已保存字段，不让持久化记忆影响工具路由。"""
    lookups = (
        ("budget", ("我的预算", "记得我的预算", "还记得我的预算"), "当前记录的预算是：{value}。"),
        ("city", ("我的城市", "之前说的城市", "记得我在哪"), "当前记录的城市是：{value}。"),
        ("interested_model", ("关注什么车型", "我的意向车型", "记得我关注"), "当前记录的关注车型是：{value}。"),
    )
    for field, markers, template in lookups:
        if any(marker in query for marker in markers):
            value = state.profile.get(field)
            return template.format(value=value) if value else "当前没有记录这项会话记忆。"
    return None


def _calculator_result_metadata(query: str, result: str) -> dict:
    """从确定性计算器的稳定输出提取轻量结构化结果。

    这里只解析计算器自己的格式，不让 LLM 参与数值判断；自然语言结果仍完整保存在
    ``calculation_result`` 中，便于回放和审计。
    """
    fields = {
        "loan_principal": r"贷款本金：\s*([\d,]+\.\d+)",
        "monthly_payment": r"月供：\s*([\d,]+\.\d+)",
        "total_interest": r"总利息：\s*([\d,]+\.\d+)",
        "total_payment": r"本息合计：\s*([\d,]+\.\d+)",
        "annual_kwh": r"年用电量：\s*([\d,]+\.\d+)",
        "annual_cost": r"年电费：\s*([\d,]+\.\d+)",
        "landed_total": r"预计落地总价：\s*([\d,]+\.\d+)",
    }
    structured = {}
    for name, pattern in fields.items():
        match = re.search(pattern, result)
        if match:
            structured[name] = float(match.group(1).replace(",", ""))
    if "月供" in result:
        calculation_type = "loan_monthly_payment"
    elif "年电费" in result:
        calculation_type = "annual_energy_cost"
    elif "预计落地总价" in result:
        calculation_type = "purchase_budget"
    else:
        calculation_type = "deterministic_calculation"
    return {
        "calculation_type": calculation_type,
        "normalized_parameters": parse_parameters(query).to_dict(),
        "result": structured,
        "calculation_result": result,
    }


def _append_calculator_evidence(trace: dict, query: str, result: str) -> None:
    evidence = make_evidence(
        f"e{len(trace['_evidence_items']) + 1}",
        "calculator",
        "deterministic_calculator",
        result,
        metadata=_calculator_result_metadata(query, result),
        validated=True,
    )
    trace["_evidence_items"].append(evidence)
    trace["tool_results"].append({"tool": "car_calculator", "result": result, "evidence_id": evidence["evidence_id"]})


def _record_failure(trace: dict, *, tool: str, stage: str, failure_type: str, message: str, recoverable: bool = True, details: dict | None = None) -> dict:
    """统一记录失败；失败信息只进入 Trace，不会伪装成 Evidence。"""
    event = {
        "failure_type": failure_type,
        "tool": tool,
        "stage": stage,
        "recoverable": recoverable,
        "message": message,
        "details": details or {},
    }
    trace["tool_error"].append(event)
    trace["failure"]["has_failure"] = True
    trace["failure"]["events"].append(event)
    if failure_type not in trace["failure"]["failure_types"]:
        trace["failure"]["failure_types"].append(failure_type)
    return event


def _safe_failure_text(events: list[dict]) -> str:
    """不调用 LLM 解释工具失败，避免工具失败后由模型补造事实。"""
    types = {event["failure_type"] for event in events}
    if "source_validation_failure" in types:
        return "未找到可验证的官方金融来源，因此不会把该结果作为金融方案或利率建议提供。"
    if "invalid_input" in types:
        return "输入参数不合法，未执行有效测算；请检查首付比例、金额和里程等参数后重试。"
    if "empty_result" in types or "no_relation" in types:
        return "当前工具未返回可用证据，因此无法据此确认相关事实。"
    if "timeout" in types:
        return "当前工具请求超时，暂时无法完成该项查询或识别，因此不会给出未经验证的结论。"
    if "planner_failure" in types or "unknown_tool" in types:
        return "当前无法可靠生成可执行工具计划，因此未执行未知工具，也不会猜测结果。"
    return "当前工具执行失败，暂时无法验证相关信息，因此不会给出未经证据支持的结论。"


def _finish_trace(state: ConversationState, trace: dict, text: str, started_at: float, answer_started_at: float | None = None) -> str:
    """统一补齐结构化 trace，同时保留旧版 last_trace 字段兼容性。"""
    now = time.perf_counter()
    trace["assistant_answer"] = text
    trace.setdefault("answer", {})["text"] = text
    trace["answer"]["success"] = True
    trace["answer"]["latency_ms"] = round((now - answer_started_at) * 1000, 2) if answer_started_at else 0.0
    trace.setdefault("performance", {})["answer_latency_ms"] = trace["answer"]["latency_ms"]
    trace["performance"]["total_latency_ms"] = round((now - started_at) * 1000, 2)
    trace["state_after"] = {"pending_tool": state.pending_tool, "slots": dict(state.slots), "profile": dict(state.profile)}
    trace["evidence"]["items"] = trace.pop("_evidence_items", [])
    if not trace["evidence"]["items"]:
        trace["evidence"]["items"] = [
            {"evidence_id": f"e{i}", "type": "local_rag", "source": item.get("entity", {}).get("source"), "title": item.get("entity", {}).get("title"), "url": None, "content": item.get("entity", {}).get("text", "")[:4000], "rank": None, "score": None, "metadata": {"chunk_index": item.get("entity", {}).get("chunk_index")}, "validated": True}
            for i, item in enumerate(trace.pop("_evidence_memory", []), start=1)
        ]
    trace["evidence"]["source_types"] = sorted({item.get("source") for item in trace["evidence"]["items"] if item.get("source")})
    trace["evidence"]["retrieved_chunks"] = sorted({item.get("chunk_index") for item in trace["evidence"]["items"] if item.get("chunk_index") is not None})
    citations = []
    for number in sorted({int(value) for value in re.findall(r"\[(\d+)\]", text)}):
        if 1 <= number <= len(trace["evidence"]["items"]):
            citations.append({"citation": f"[{number}]", "evidence_id": trace["evidence"]["items"][number - 1]["evidence_id"]})
    trace["answer"]["citations"] = citations
    claim_evidence = []
    calculator_items = [item for item in trace["evidence"]["items"] if item.get("type") == "calculator"]
    if calculator_items:
        # 确定性计算器的回答就是工具原文；按回答行与原文行建立可审计映射，
        # 不向用户强行插入 [1]。
        for line in text.splitlines():
            claim = line.strip()
            if claim and claim in calculator_items[0].get("content", ""):
                claim_evidence.append({"claim": claim, "evidence_id": calculator_items[0]["evidence_id"]})
    trace["answer"]["claim_evidence"] = claim_evidence
    trace["answer"]["citation_mapping_status"] = (
        "complete" if citations or claim_evidence else
        ("not_applicable" if not trace["evidence"]["items"] else "none")
    )
    successful = {call["tool"] for call in trace.get("execution", {}).get("tool_calls", []) if call.get("success")}
    failed = trace["failure"].get("events", [])
    if failed and successful and "partial_tool_failure" not in trace["failure"]["failure_types"]:
        trace["failure"]["failure_types"].append("partial_tool_failure")
    memory_trace = trace.setdefault("memory", {
        "session_id": state.session_id,
        "loaded": state.memory_loaded,
        "loaded_fields": list(state.memory_loaded_fields),
        "updated_fields": list(state.memory_updated_fields),
        "persisted": False,
    })
    if state.memory_error:
        memory_trace["error"] = state.memory_error
        _record_failure(trace, tool="memory", stage="persistence", failure_type="memory_failure", message=state.memory_error)
    elif state.session_id and state.memory_manager:
        try:
            state.memory_manager.save_memory(state.session_id, state)
            memory_trace["persisted"] = True
        except MemoryStoreError as error:
            memory_trace["error"] = str(error)
            _record_failure(trace, tool="memory", stage="persistence", failure_type="memory_failure", message=str(error))
    state.memory_updated_fields = []
    state.last_trace = trace
    return text


LOAN_SLOTS = ("car_price", "down_payment", "years", "annual_rate")
LOAN_PROMPTS = {
    "car_price": "可以。请先告诉我裸车价，例如：30 万。",
    "down_payment": "首付比例是多少？例如：30%。",
    "years": "计划贷款几年？例如：3 年。",
    "annual_rate": "年利率是多少？例如：3.5%。",
}


def _fill_loan_slots(text: str, state: ConversationState) -> None:
    """优先识别带字段名的参数；单独回答时按当前待填槽位解释。"""
    patterns = {
        "car_price": r"(?:车价|裸车价|价格)?\s*(\d+(?:\.\d+)?)\s*万",
        "down_payment": r"(?:首付(?:比例)?)\s*(\d+(?:\.\d+)?)\s*%",
        "years": r"(?:贷款(?:期限)?|分期)?\s*(\d+(?:\.\d+)?)\s*年",
        "annual_rate": r"(?:年利率|利率|按|按照)\s*(\d+(?:\.\d+)?)\s*%",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            state.slots[name] = match.group(1)

    # 统一复用参数解析器处理“首付三成”等口语；槽位内部保持百分数形式，
    # 这样后续拼接给 calculator 时不会把 0.3 当成 0.3%。
    if "down_payment" not in state.slots:
        parsed_down = parse_parameters(text).down_payment_ratio
        if parsed_down is not None:
            state.slots["down_payment"] = str(round(parsed_down * 100, 6)).rstrip("0").rstrip(".")

    missing = next((name for name in LOAN_SLOTS if name not in state.slots), None)
    if not missing:
        return
    fallback = {
        "car_price": r"(\d+(?:\.\d+)?)\s*万",
        "down_payment": r"(\d+(?:\.\d+)?)\s*%",
        "years": r"(\d+(?:\.\d+)?)\s*年",
        "annual_rate": r"(\d+(?:\.\d+)?)\s*%",
    }[missing]
    if missing == "annual_rate" and not re.search(r"(?:年利率|利率|按|按照)\s*\d+(?:\.\d+)?\s*%?", text):
        # 在已经进入年利率槽位后，单独回复“3.5%”是合法的；避免把首付百分比当利率。
        if "首付" in text or not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*%\s*[。.!！]??\s*", text):
            return
    # 不要从“利率 3.5%”中把百分号误当成首付比例。
    if missing == "down_payment" and "首付" not in text and not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*%\s*[。.!！]?\s*", text):
        return
    match = re.search(fallback, text)
    if match:
        state.slots[missing] = match.group(1)


def _loan_dialogue(user_query: str, state: ConversationState) -> str | None:
    finance_terms = ("工行", "工商银行", "建行", "建设银行", "中行", "农行", "交行", "招行", "分期方案", "金融方案", "官方方案")
    if state.pending_tool is None and len(state.recent_turns) >= 2 and "取消" in state.recent_turns[-2] and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:年|%)?.*", user_query):
        return "本次贷款测算已取消；如需重新开始，请重新提供车价、首付、期限和利率。"
    is_finance_lookup = any(word in user_query for word in ("查询", "联网", "查一下", "查", "哪些", "适合我", "比较", "方案", "暂不计算")) and any(word in user_query for word in finance_terms)
    if is_finance_lookup and "适合我" in user_query and not state.profile.get("city") and not any(word in user_query for word in finance_terms[:7]):
        state.pending_tool = "finance"
        return "可以帮你查官方金融方案。请先告诉我所在城市，以及想比较的银行（例如工行、建行）；如果还要计算月供，再提供车价、首付比例和贷款年限。"
    if state.pending_tool == "finance" and is_finance_lookup:
        return None
    # 明确的官方金融查询优先级高于尚未完成的贷款槽位，避免把“查方案”误当成继续填参数。
    if is_finance_lookup:
        return None
    # 仅表达“想贷款”属于画像偏好，不应立即启动槽位追问；用户明确要算月供时再启动。
    if state.pending_tool != "loan" and "贷款" in user_query and not any(word in user_query for word in ("帮我算", "计算", "月供", "首付", "车价", "裸车价", "贷款几年", "利率")):
        return None
    if state.pending_tool != "loan" and not any(word in user_query for word in ("贷款", "月供", "首付")):
        return None
    if user_query.strip() in {"取消", "取消计算", "退出计算"} or ("取消" in user_query and any(word in user_query for word in ("贷款", "计算", "月供"))):
        state.pending_tool, state.slots = None, {}
        return "已取消本次贷款测算。"
    if any(word in user_query for word in ("先别算", "先不算", "暂不计算", "暂时不算", "不要计算", "先不要计算")):
        # 暂停只停止本轮执行，保留 pending_tool 与已填槽位，便于用户稍后继续。
        state.pending_tool = "loan"
        return "好的，暂不计算。已保留目前参数，确认年利率或贷款期限后，再继续贷款测算。"
    state.pending_tool = "loan"
    _fill_loan_slots(user_query, state)
    missing = next((name for name in LOAN_SLOTS if name not in state.slots), None)
    if missing:
        if missing == "annual_rate" and any(word in user_query for word in ("查询", "联网", "查一下")):
            try:
                result = finance_search(user_query)[0]["entity"]["text"]
            except RuntimeError as error:
                return f"官方金融方案检索暂不可用：{error}"
            return result + "\n\n以上为实时官方页面参考，不构成授信承诺。请确认一个明确年利率（例如“按 3.5% 算”）后，我再计算月供。"
        if missing == "annual_rate" and any(word in user_query for word in ("忘记", "一般", "多少", "不知道", "不清楚")):
            return "实际年利率会因银行、金融方案、活动和个人资质而变化，不能替你假定最终报价。若要联网查询，请提供城市、意向车型和目标机构，例如“查询杭州 ES9 的工行、建行汽车金融方案”；或回复“按 3.5% 算”进行示例测算。"
        return LOAN_PROMPTS[missing]

    slots = state.slots
    query = f"车价{slots['car_price']}万，首付{slots['down_payment']}%，贷款{slots['years']}年，年利率{slots['annual_rate']}%，月供多少？"
    state.pending_tool, state.slots = None, {}
    return calculate_from_query(query)


def run(user_query: str, state: ConversationState | None = None) -> str:
    state = state or ConversationState()
    started_at = time.perf_counter()
    pending_before = state.pending_tool
    trace = {
        "trace_id": uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": user_query,
        "planned_tools": [],
        "executed_tools": [],
        "tool_results": [],
        "tool_success": [],
        "tool_error": [],
        "state_before": {"pending_tool": state.pending_tool, "slots": dict(state.slots), "profile": dict(state.profile)},
        "planner": {"planned_tools": [], "raw_plan": None, "latency_ms": None, "success": True, "error": None},
        "execution": {"executed_tools": [], "tool_calls": []},
        "failure": {"has_failure": False, "failure_types": [], "events": []},
        "memory": {
            "session_id": state.session_id,
            "loaded": state.memory_loaded,
            "loaded_fields": list(state.memory_loaded_fields),
            "updated_fields": [],
            "persisted": False,
        },
        "evidence": {"items": [], "source_types": [], "retrieved_chunks": []},
        "answer": {"text": "", "latency_ms": None, "success": True, "error": None},
        "performance": {"planner_latency_ms": None, "tool_latency_ms": 0.0, "answer_latency_ms": None, "total_latency_ms": None},
        "_evidence_memory": [],
        "_evidence_items": [],
    }
    state.last_trace = trace
    profile_before = dict(state.profile)
    _update_profile(user_query, state)
    state.memory_updated_fields = sorted(key for key, value in state.profile.items() if profile_before.get(key) != value)
    trace["memory"]["updated_fields"] = list(state.memory_updated_fields)
    memory_lookup = _memory_lookup_response(user_query, state)
    if memory_lookup is not None:
        trace["behavior"] = "memory_lookup"
        return _finish_trace(state, trace, memory_lookup, started_at)
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:万)?\s*左右[。！!]??", user_query) and any(word in " ".join(state.recent_turns) for word in ("银行", "方案", "金融")):
        amount = re.search(r"\d+(?:\.\d+)?", user_query).group(0)
        state.profile["budget"] = amount + "万"
        return _finish_trace(state, trace, f"已记录预算约 {amount} 万元。若要计算月供，还需要首付比例、贷款年限和确认后的年利率。", started_at)
    state.recent_turns.append(user_query)
    state.recent_turns[:] = state.recent_turns[-10:]
    loan_intent = any(word in user_query for word in ("贷款", "月供", "首付", "贷", "分期"))
    product_intent = any(word in user_query for word in ("ES9", "星辰", "充电", "续航", "座舱", "驾驶", "配置", "电池", "问界", "小鹏", "理想", "蔚来", "特斯拉"))
    mixed_request = loan_intent and product_intent
    # 混合任务不能由贷款槽位对话提前终止；先保留槽位，再让 Planner 执行其余任务。
    if mixed_request:
        state.pending_tool = "loan"
        _fill_loan_slots(user_query, state)
        dialogue_result = None
    else:
        dialogue_result = _loan_dialogue(user_query, state)
    if dialogue_result is not None:
        calculator_injection = injected_failure(state.failure_injection, "car_calculator")
        if calculator_injection and (loan_intent or pending_before == "loan"):
            trace["planned_tools"] = ["car_calculator"]
            trace["planner"]["planned_tools"] = ["car_calculator"]
            _record_failure(trace, tool="car_calculator", stage="execution", **calculator_injection)
            trace["execution"]["tool_calls"].append({"tool": "car_calculator", "input": {"query": user_query}, "output": None, "success": False, "failure_type": calculator_injection["failure_type"], "error": calculator_injection["message"], "recoverable": calculator_injection["recoverable"], "latency_ms": 0.0})
            trace["behavior"] = "safe_failure"
            return _finish_trace(state, trace, _safe_failure_text(trace["failure"]["events"]), started_at)
        if (loan_intent or pending_before == "loan") and "已取消" not in dialogue_result and not any(word in user_query for word in ("先别算", "先不算", "暂不计算", "暂时不算", "不要计算", "先不要计算")):
            trace["planned_tools"] = ["car_calculator"]
            if "贷款测算" in dialogue_result or "月供：" in dialogue_result:
                trace["executed_tools"] = ["car_calculator"]
                trace["tool_success"] = ["car_calculator"]
                _append_calculator_evidence(trace, user_query, dialogue_result)
            elif dialogue_result.startswith("计算工具需要补充参数："):
                _record_failure(trace, tool="car_calculator", stage="validation", failure_type="invalid_input", message=dialogue_result)
        trace["behavior"] = "slot_dialogue" if state.pending_tool else "dialogue_response"
        trace["planner"]["latency_ms"] = 0.0
        trace["performance"]["planner_latency_ms"] = 0.0
        return _finish_trace(state, trace, dialogue_result, started_at)

    planner_started_at = time.perf_counter()
    planner_injection = injected_failure(state.failure_injection, "planner")
    if planner_injection:
        trace["planner"].update({"success": False, "error": planner_injection["message"], "failure_type": planner_injection["failure_type"]})
        trace["planner"]["latency_ms"] = round((time.perf_counter() - planner_started_at) * 1000, 2)
        trace["performance"]["planner_latency_ms"] = trace["planner"]["latency_ms"]
        _record_failure(trace, tool="planner", stage="planning", **planner_injection)
        trace["behavior"] = "safe_failure"
        return _finish_trace(state, trace, _safe_failure_text(trace["failure"]["events"]), started_at)
    actions = plan(user_query)
    planner_latency = round((time.perf_counter() - planner_started_at) * 1000, 2)
    trace["planner"]["latency_ms"] = planner_latency
    trace["performance"]["planner_latency_ms"] = planner_latency
    trace["planned_tools"] = [action.tool for action in actions]
    trace["planner"]["planned_tools"] = list(trace["planned_tools"])
    if not actions:
        vision_injection = injected_failure(state.failure_injection, "vision")
        if vision_injection:
            _record_failure(trace, tool="vision", stage="execution", **vision_injection)
            trace["execution"]["tool_calls"].append({"tool": "vision", "input": {"query": user_query}, "output": None, "success": False, "failure_type": vision_injection["failure_type"], "error": vision_injection["message"], "recoverable": vision_injection["recoverable"], "latency_ms": 0.0})
            trace["behavior"] = "safe_failure"
            return _finish_trace(state, trace, _safe_failure_text(trace["failure"]["events"]), started_at)
        result = "好的，已记录你的偏好。需要查询产品资料、实时信息或进行计算时，我再调用对应工具。"
        trace["behavior"] = "no_tool"
        return _finish_trace(state, trace, result, started_at)
    print("\n[1/3] 规划结果：")
    for action in actions:
        print(f"- {action.tool}: {action.query}（{action.reason}）")

    print("[2/3] 执行工具…")
    memory: list[dict] = []
    graph_injection = injected_failure(state.failure_injection, "graph")
    if graph_injection:
        _record_failure(trace, tool="graph", stage="execution", **graph_injection)
        trace["execution"]["tool_calls"].append({"tool": "graph", "input": {"query": user_query}, "output": None, "success": False, "failure_type": graph_injection["failure_type"], "error": graph_injection["message"], "recoverable": graph_injection["recoverable"], "latency_ms": 0.0})
    else:
        try:
            graph_text = JsonGraph().context_for(user_query)
            if graph_text:
                print("[Graph] 正在查询 JSON 知识图谱…", flush=True)
                memory.append({"entity": {"text": graph_text, "source": "JSON知识图谱", "title": "结构化关系", "chunk_index": 0}, "distance": 0.0})
                trace["_evidence_items"].append(make_evidence(f"e{len(trace['_evidence_items']) + 1}", "graph", "knowledge_graph", graph_text, metadata={"query": user_query}, validated=True))
        except (FileNotFoundError, ValueError, KeyError) as error:
            _record_failure(trace, tool="graph", stage="execution", failure_type="execution_failure", message=str(error))
    vision_injection = injected_failure(state.failure_injection, "vision")
    if vision_injection:
        _record_failure(trace, tool="vision", stage="execution", **vision_injection)
        trace["execution"]["tool_calls"].append({"tool": "vision", "input": {"query": user_query}, "output": None, "success": False, "failure_type": vision_injection["failure_type"], "error": vision_injection["message"], "recoverable": vision_injection["recoverable"], "latency_ms": 0.0})
    else:
        memory.extend({"entity": {"text": text, "source": "图片识别", "title": "本次会话图片识别结果", "chunk_index": 0}, "distance": 0.0} for text in state.vision_memory)
        for text in state.vision_memory:
            trace["_evidence_items"].append(make_evidence(f"e{len(trace['_evidence_items']) + 1}", "vision", "qwen_vl_plus", text, metadata={"model": settings.vision_model}, validated=True))
    calculation_results: list[str] = []
    finance_results: list[str] = []
    missing_slot_prompt: str | None = None
    for action in actions:
        tool_started_at = time.perf_counter()
        tool_call = {"tool": action.tool, "input": {"query": action.query}, "output": None, "success": False, "failure_type": None, "latency_ms": None, "error": None, "recoverable": False}
        trace["executed_tools"].append(action.tool)
        trace["execution"]["executed_tools"].append(action.tool)
        injected = injected_failure(state.failure_injection, action.tool)
        if injected:
            _record_failure(trace, tool=action.tool, stage="execution", **injected)
            tool_call.update({"failure_type": injected["failure_type"], "error": injected["message"], "recoverable": injected["recoverable"], "latency_ms": round((time.perf_counter() - tool_started_at) * 1000, 2)})
            trace["execution"]["tool_calls"].append(tool_call)
            continue
        if action.tool not in {"local_rag", "web_search", "finance_web_search", "car_calculator"}:
            _record_failure(trace, tool=action.tool, stage="execution", failure_type="unknown_tool", message=f"不支持的工具：{action.tool}", recoverable=False)
            tool_call.update({"failure_type": "unknown_tool", "error": f"不支持的工具：{action.tool}", "recoverable": False, "latency_ms": round((time.perf_counter() - tool_started_at) * 1000, 2)})
            trace["execution"]["tool_calls"].append(tool_call)
            continue
        if action.tool == "local_rag":
            embedder = get_embedder(
                settings.embedding_backend,
                settings.embedding_model,
                settings.embedding_base_url,
                settings.embedding_dimensions,
                settings.openai_api_key,
            )
            from vector_store import LocalVectorStore

            store = LocalVectorStore(settings.resolved_vector_store_path(), settings.collection_name, embedder.dimension)
            try:
                hits = store.search(
                    embedder.embed_query(action.query), query_text=action.query,
                    ranking_strategy=settings.ranking_strategy,
                    semantic_weight=settings.semantic_weight,
                )
                if not hits:
                    _record_failure(trace, tool=action.tool, stage="execution", failure_type="empty_result", message="本地知识库未返回检索结果")
                    tool_call.update({"failure_type": "empty_result", "error": "本地知识库未返回检索结果"})
                    tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
                    trace["execution"]["tool_calls"].append(tool_call)
                    continue
                trace["tool_success"].append(action.tool)
                trace["tool_results"].append({"tool": action.tool, "count": len(hits)})
                for hit in hits:
                    entity = hit["entity"]
                    trace["_evidence_items"].append(make_evidence(
                        f"e{len(trace['_evidence_items']) + 1}", "local_rag", "local_knowledge_base", entity.get("text", ""),
                        title=entity.get("title"), rank=hit.get("rank"), score=hit.get("final_score"),
                        metadata={"chunk_index": entity.get("chunk_index"), "semantic_score": hit.get("semantic_score"), "lexical_score": hit.get("lexical_score"), "final_score": hit.get("final_score"), "top_k": 4}, validated=True,
                    ))
                memory.extend(hits)
                tool_call["output"] = {"count": len(hits), "chunks": [h["entity"].get("chunk_index") for h in hits]}
                tool_call["success"] = True
            except Exception as error:
                _record_failure(trace, tool=action.tool, stage="execution", failure_type="retrieval_failure", message=str(error))
                tool_call.update({"failure_type": "retrieval_failure", "error": str(error)})
        elif action.tool == "web_search":
            try:
                hits = web_search(action.query)
                if not hits or not hits[0].get("entity", {}).get("text", "").strip():
                    _record_failure(trace, tool=action.tool, stage="execution", failure_type="empty_result", message="联网搜索未返回可用结果")
                    tool_call.update({"failure_type": "empty_result", "error": "联网搜索未返回可用结果"})
                    tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
                    trace["execution"]["tool_calls"].append(tool_call)
                    continue
                trace["tool_success"].append(action.tool)
                for rank, hit in enumerate(hits, start=1):
                    entity = hit.get("entity", {})
                    trace["_evidence_items"].append(make_evidence(f"e{len(trace['_evidence_items']) + 1}", "web", "bailian_web_search", entity.get("text", ""), title=entity.get("title"), url=entity.get("url"), rank=rank, metadata={"query": action.query}, validated=True))
                memory.extend(hits)
                tool_call["output"] = {"count": len(hits), "sources": [h["entity"].get("source") for h in hits]}
                tool_call["success"] = True
            except RuntimeError as error:
                failure_type = "timeout" if "超时" in str(error) else "execution_failure"
                _record_failure(trace, tool=action.tool, stage="execution", failure_type=failure_type, message=str(error))
                tool_call.update({"failure_type": failure_type, "error": str(error)})
                # ponytail: 网页工具是增强项；不可用时保留本地 RAG 的可用性。
                print(f"[Web] 网页搜索不可用，已跳过：{error}", flush=True)
        elif action.tool == "finance_web_search":
            try:
                hits = finance_search(action.query)
                if not hits or not hits[0].get("entity", {}).get("text", "").strip():
                    _record_failure(trace, tool=action.tool, stage="execution", failure_type="empty_result", message="金融检索未返回结果")
                    tool_call.update({"failure_type": "empty_result", "error": "金融检索未返回结果"})
                    tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
                    trace["execution"]["tool_calls"].append(tool_call)
                    continue
                result = hits[0]["entity"]["text"]
                entity = hits[0].get("entity", {})
                official = "http" in result and any(site in result for site in ("icbc.com.cn", "ccb.com", "boc.cn", "abchina.com", "bankcomm.com", "cmbchina.com"))
                if not official:
                    _record_failure(trace, tool=action.tool, stage="source_validation", failure_type="source_validation_failure", message="金融结果缺少可验证银行官网链接")
                    tool_call.update({"failure_type": "source_validation_failure", "error": "金融结果缺少可验证银行官网链接", "output": {"count": len(hits), "official_source_found": False}})
                    tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
                    trace["execution"]["tool_calls"].append(tool_call)
                    continue
                finance_results.append(result)
                trace["tool_success"].append(action.tool)
                trace["_evidence_items"].append(make_evidence(f"e{len(trace['_evidence_items']) + 1}", "finance", "bank_official_site" if official else "bailian_finance_search", result, title=entity.get("title"), url=entity.get("url"), rank=1, metadata={"official_source_verified": official, "validation": "passed" if official else "failed"}, validated=official))
                memory.extend(hits)
                tool_call["output"] = {"count": len(hits), "official_source_found": "http" in result, "source_preview": result[:240]}
                tool_call["success"] = True
            except RuntimeError as error:
                failure_type = "timeout" if "超时" in str(error) else "execution_failure"
                _record_failure(trace, tool=action.tool, stage="execution", failure_type=failure_type, message=str(error))
                tool_call.update({"failure_type": failure_type, "error": str(error)})
                print(f"[Finance] 官方金融方案检索不可用，已跳过：{error}", flush=True)
        elif action.tool == "car_calculator":
            print("[Calculator] 正在执行确定性购车测算…", flush=True)
            missing_names = missing_calculation_parameters(user_query, state.slots if loan_intent else None)
            if missing_names:
                missing_slot_prompt = LOAN_PROMPTS[missing_names[0]] if len(missing_names) == 1 and missing_names[0] in LOAN_PROMPTS else missing_parameter_prompt(missing_names)
                tool_call["output"] = {"missing_parameters": missing_names}
                # 这是 planned 但未 executed 的工具，不把缺参请求记为计算成功。
                trace["executed_tools"].remove(action.tool)
                trace["execution"]["executed_tools"].remove(action.tool)
                tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
                trace["execution"]["tool_calls"].append(tool_call)
                continue
            # 计算参数必须读取用户原句；规划模型可能会把“18 度/100 公里”缩写为“18 度”。
            result = calculate_from_query(user_query)
            if result.startswith("计算工具需要补充参数："):
                _record_failure(trace, tool=action.tool, stage="validation", failure_type="invalid_input", message=result)
                tool_call.update({"failure_type": "invalid_input", "error": result, "output": {"validation_message": result}})
                missing_slot_prompt = result
                trace["executed_tools"].remove(action.tool)
                trace["execution"]["executed_tools"].remove(action.tool)
                tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
                trace["execution"]["tool_calls"].append(tool_call)
                continue
            calculation_results.append(result)
            if mixed_request and "贷款测算" in result:
                state.pending_tool, state.slots = None, {}
            trace["tool_success"].append(action.tool)
            trace["tool_results"].append({"tool": action.tool, "result": result})
            tool_call["output"] = {"normalized_parameters": parse_parameters(user_query).to_dict(), "result": result}
            tool_call["success"] = True
            _append_calculator_evidence(trace, user_query, result)
            memory.append(
                {
                    "entity": {
                        "text": result,
                        "source": "购车计算工具",
                        "chunk_index": 0,
                        "title": "确定性测算结果",
                    },
                    "distance": 0.0,
                }
            )
        tool_call["latency_ms"] = round((time.perf_counter() - tool_started_at) * 1000, 2)
        trace["performance"]["tool_latency_ms"] += tool_call["latency_ms"]
        trace["execution"]["tool_calls"].append(tool_call)

    failures = trace["failure"]["events"]
    successful_tools = set(trace["tool_success"])
    # 部分失败时优先保留确定性计算结果；失败工具不交给 LLM“补答”。
    if calculation_results and failures:
        trace["behavior"] = "partial_failure_recovery"
        trace["_evidence_memory"] = memory
        return _finish_trace(state, trace, "\n\n".join(calculation_results + [_safe_failure_text(failures)]), started_at)
    # 所有实际工具都失败时，直接安全失败，不向 Answer Generator 传递空上下文。
    if failures and not successful_tools:
        trace["behavior"] = "safe_failure"
        trace["_evidence_memory"] = memory
        return _finish_trace(state, trace, _safe_failure_text(failures), started_at)

    # ponytail: 纯数学题的正确答案已由确定性工具生成，禁止 LLM 二次改写。
    if calculation_results and all(action.tool == "car_calculator" for action in actions):
        print("[3/3] 返回计算工具结果…\n")
        trace["_evidence_memory"] = memory
        return _finish_trace(state, trace, "\n\n".join(calculation_results), started_at)
    if missing_slot_prompt and all(action.tool == "car_calculator" for action in actions):
        # 纯计算请求缺参时直接追问，不再调用回答模型，也不把缺参文本误算成工具结果。
        trace["behavior"] = "request_missing_slots"
        trace["_evidence_memory"] = memory
        return _finish_trace(state, trace, missing_slot_prompt, started_at)
    if finance_results and all(action.tool == "finance_web_search" for action in actions):
        print("[3/3] 返回官方金融检索结果…\n")
        profile_text = _profile_memory(state)
        if state.pending_tool == "finance":
            state.pending_tool = None
        context = f"已结合本次会话偏好：{profile_text}\n\n" if profile_text else ""
        result = context + "\n\n".join(finance_results) + "\n\n以上为实时官方页面参考，不构成授信承诺；请确认具体年利率后再进行月供测算。"
        trace["_evidence_memory"] = memory
        return _finish_trace(state, trace, result, started_at)

    print("[3/3] 根据检索证据生成回答…\n")
    profile_text = _profile_memory(state)
    if profile_text:
        memory.append({"entity": {"text": f"用户主动提供的会话偏好：{profile_text}", "source": "会话记忆", "title": "当前会话记忆"}, "distance": 0.0})
    answer_started_at = time.perf_counter()
    try:
        result = answer(user_query, memory)
    except Exception as error:
        trace["answer"] = {"text": "", "latency_ms": round((time.perf_counter() - answer_started_at) * 1000, 2), "success": False, "error": {"type": type(error).__name__, "message": str(error)}}
        trace["performance"]["answer_latency_ms"] = trace["answer"]["latency_ms"]
        trace["performance"]["total_latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        trace["state_after"] = {"pending_tool": state.pending_tool, "slots": dict(state.slots), "profile": dict(state.profile)}
        state.last_trace = trace
        raise
    if missing_slot_prompt:
        result = result + "\n\n" + missing_slot_prompt
    if failures:
        result = result + "\n\n" + _safe_failure_text(failures)
    trace["_evidence_memory"] = memory
    return _finish_trace(state, trace, result, started_at, answer_started_at)
