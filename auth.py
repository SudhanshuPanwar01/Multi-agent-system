import hashlib
import json
import os

USERS_FILE = "users.json"


def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def sign_up(username: str, password: str):
    users = _load_users()
    if not username.strip():
        return False, "Username cannot be empty."
    if username in users:
        return False, "Username already exists."
    users[username] = _hash(password)
    _save_users(users)
    return True, "Account created! Please log in."


def login(username: str, password: str):
    users = _load_users()
    if username not in users:
        return False, "User not found. Please sign up first."
    if users[username] != _hash(password):
        return False, "Wrong password."
    return True, "Login successful!"