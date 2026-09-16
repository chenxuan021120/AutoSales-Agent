"""JSON 知识图谱的最小查询接口。"""
from __future__ import annotations

import json
from pathlib import Path


class JsonGraph:
    def __init__(self, path: str | Path = "data/knowledge_graph.json") -> None:
        # 相对路径必须相对于项目目录，而不是当前 shell 的工作目录。
        # 这样从项目外执行绝对路径 app.py 时也能正确找到 data/knowledge_graph.json。
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent / candidate
        self.path = candidate.resolve()
        if not self.path.exists():
            raise FileNotFoundError(
                f"知识图谱文件不存在：{self.path}。请确认项目目录下有 data/knowledge_graph.json。"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in payload["nodes"]}
        self.edges = payload["edges"]

    def neighbors(self, node_id: str, relation: str | None = None) -> list[dict]:
        return [
            {"node": self.nodes[edge["target"]], "relation": edge["relation"], "condition": edge.get("condition")}
            for edge in self.edges
            if edge["source"] == node_id and (relation is None or edge["relation"] == relation)
        ]

    def find_by_name(self, keyword: str) -> list[dict]:
        return [node for node in self.nodes.values() if keyword.lower() in node["name"].lower()]

    def context_for(self, question: str) -> str:
        """按问题主题返回一小段结构化关系，供回答器与 RAG 并列参考。"""
        return "\n".join(item["content"] for item in self.relations_for(question))

    def relations_for(self, question: str) -> list[dict]:
        """返回结构化关系，供 Evidence layer 一条关系对应一个 evidence item。"""
        lines = []
        topics = []
        if "充电" in question:
            topics.append("charging")
        if "座舱" in question or "屏幕" in question:
            topics.append("cockpit")
        if "驾驶" in question or "智驾" in question:
            topics.append("driving")
        needs = []
        if any(word in question for word in ("长途", "高速", "续航焦虑")):
            needs.append("need_long_trip")
        if any(word in question for word in ("家庭", "孩子", "后排")):
            needs.append("need_family")
        if any(word in question for word in ("智能驾驶", "智驾", "辅助驾驶")):
            needs.append("need_smart_drive")
        if any(word in question for word in ("预算", "省钱", "性价比")):
            needs.append("need_budget")
        for need in needs:
            for item in self.neighbors(need, "PREFERS"):
                lines.append({"content": f"用户需求 {self.nodes[need]['name']} —PREFERS→ {item['node']['name']}", "source_node": self.nodes[need]["name"], "relation": "PREFERS", "target_node": item["node"]["name"], "condition": item.get("condition")})
        for topic in topics:
            topic_name = self.nodes[topic]["name"]
            lines.append({"content": f"星辰电动 ES9 —HAS_FEATURE→ {topic_name}", "source_node": "星辰电动 ES9", "relation": "HAS_FEATURE", "target_node": topic_name, "condition": None})
            for item in self.neighbors(topic):
                condition = f"（{item['condition']}）" if item.get("condition") else ""
                lines.append({"content": f"{topic_name} —{item['relation']}→ {item['node']['name']}{condition}", "source_node": topic_name, "relation": item["relation"], "target_node": item["node"]["name"], "condition": item.get("condition")})
                for child in self.neighbors(item["node"]["id"]):
                    lines.append({"content": f"{item['node']['name']} —{child['relation']}→ {child['node']['name']}", "source_node": item["node"]["name"], "relation": child["relation"], "target_node": child["node"]["name"], "condition": child.get("condition")})
        return lines


if __name__ == "__main__":
    graph = JsonGraph()
    for item in graph.neighbors("es9"):
        print(item["relation"], "->", item["node"]["name"])
