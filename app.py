"""PyCharm 中直接右键运行此文件，或在 Terminal 使用命令。"""
from __future__ import annotations

import argparse

from agent import ConversationState, clear_session_memory, load_session_state, run
from ingest import build_index
from vision import analyze_image


def chat(session_id: str | None = None) -> None:
    print("ES9 Agent 已启动。输入问题；输入 q 退出。")
    state = load_session_state(session_id) if session_id else ConversationState()
    if session_id:
        print(f"已加载会话：{session_id}")
    while True:
        question = input("\n你：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            return
        if question == "/memory":
            label = "当前持久化记忆" if state.session_id else "当前会话画像（未持久化）"
            print(f"\n{label}：{state.profile or '暂无'}")
            print(f"最近对话：{state.recent_turns or '暂无'}")
            print(f"图片记忆：{len(state.vision_memory)} 条")
            continue
        if question == "/clear_memory":
            if state.session_id:
                try:
                    clear_session_memory(state)
                    print("\n已清除当前 Session 的持久化记忆。")
                except RuntimeError as error:
                    print(f"\n清除持久化记忆失败：{error}")
            else:
                state = ConversationState()
                print("\n已清除当前运行时会话记忆。")
            continue
        if question == "/clear_vision":
            state.vision_memory.clear()
            print("\n已清除图片识别记忆。")
            continue
        if question.startswith("/vision "):
            image_path = question.split(" ", 1)[1].strip()
            try:
                result = analyze_image(image_path)
                state.vision_memory.append(result)
                # 只保存有限摘要，不把完整多模态原文写入 SQLite。
                state.vision_summary = result.replace("\n", " ")[:300]
                print("\n图片识别结果已保存到当前会话记忆。")
                print(result)
            except (FileNotFoundError, RuntimeError) as error:
                print(f"\n图片识别失败：{error}")
            continue
        if question:
            print("\n助手：" + run(question, state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "chat", "vision"], help="先 ingest 建库，再 chat 对话；vision 分析车型图片")
    parser.add_argument("image", nargs="?", help="vision 命令使用的图片路径")
    parser.add_argument("--session", help="chat 使用的持久化 Session ID；省略则保持仅本次运行的临时会话")
    parser.add_argument("--question", default="识别图片中的车型、配置参数和价格；看不清的内容明确标注为无法确认。")
    args = parser.parse_args()
    if args.command == "ingest":
        build_index()
    elif args.command == "chat":
        chat(args.session)
    else:
        if not args.image:
            parser.error("vision 命令需要图片路径")
        print(analyze_image(args.image, args.question))
