from types import SimpleNamespace
import json

from scripts import run_bird_holdout


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)

    def json(self):
        return self._payload


def test_openai_generator_posts_chat_completions(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return DummyResponse({
            "choices": [
                {
                    "message": {
                        "content": '[{"question":"ping"}]',
                        "reasoning_content": "hidden",
                    }
                }
            ]
        })

    monkeypatch.setattr(run_bird_holdout.requests, "post", fake_post)

    args = SimpleNamespace(
        generator_provider="openai",
        generator_base_url="https://api.deepseek.com",
        generator_api_key="secret",
        generator_model="deepseek-v4-pro",
        generator_max_tokens=64,
        generator_temperature=0.2,
    )

    result = run_bird_holdout.call_generator(args, "Return JSON.")

    assert result == '[{"question":"ping"}]'
    assert calls[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert calls[0]["json"]["model"] == "deepseek-v4-pro"


def test_anthropic_generator_keeps_messages_endpoint(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return DummyResponse({"content": [{"type": "text", "text": '[{"question":"pong"}]'}]})

    monkeypatch.setattr(run_bird_holdout.requests, "post", fake_post)

    args = SimpleNamespace(
        generator_provider="anthropic",
        claude_url="http://127.0.0.1:8317/v1/messages",
        claude_api_key="secret",
        claude_model="claude-opus-4-6[1m]",
        generator_max_tokens=64,
        generator_temperature=0.0,
    )

    result = run_bird_holdout.call_generator(args, "Return JSON.")

    assert result == '[{"question":"pong"}]'
    assert calls[0]["url"] == "http://127.0.0.1:8317/v1/messages"
    assert calls[0]["headers"]["x-api-key"] == "secret"


def test_collect_refined_targets_reads_changed_columns(tmp_path):
    db_dir = tmp_path / "financial"
    db_dir.mkdir()
    (db_dir / "refined_tables.json").write_text(json.dumps({
        "db_id": "financial",
        "tables": [
            {
                "name": "account",
                "columns": [
                    {"name": "account_id", "original_name": "account_id"},
                    {"name": "transaction_frequency", "original_name": "freq"},
                ],
            }
        ],
    }))

    targets = run_bird_holdout.collect_refined_targets(tmp_path)

    assert len(targets) == 1
    assert targets[0]["db_id"] == "financial"
    assert targets[0]["table"] == "account"
    assert targets[0]["original_column"] == "freq"
    assert targets[0]["refined_column"] == "transaction_frequency"


def test_target_column_touch_detection_handles_quoted_names():
    assert run_bird_holdout.sql_touches_column(
        'SELECT "Free Meal Count (K-12)" FROM frpm',
        "Free Meal Count (K-12)",
    )
    assert not run_bird_holdout.sql_touches_column(
        'SELECT "FRPM Count (K-12)" FROM frpm',
        "Free Meal Count (K-12)",
    )
