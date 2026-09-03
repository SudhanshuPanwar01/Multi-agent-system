import json
import os
import time

HISTORY_DIR = "history"


def save_report(username: str, topic: str, report: str, feedback: str):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    entry = {
        "user": username,
        "topic": topic,
        "report": report,
        "feedback": feedback,
        "time": time.strftime("%Y-%m-%d %H:%M"),
    }
    filename = f"{int(time.time())}.json"
    with open(os.path.join(HISTORY_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)


def load_history(username: str) -> list:
    if not os.path.exists(HISTORY_DIR):
        return []
    items = []
    for fn in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(HISTORY_DIR, fn), "r", encoding="utf-8") as f:
            entry = json.load(f)
        if entry.get("user") == username:
            items.append(entry)
    return items