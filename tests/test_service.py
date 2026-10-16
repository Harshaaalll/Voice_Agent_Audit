import pytest
from fastapi.testclient import TestClient

from voice_agent_audit.service.app import create_app

ACCT = {"loan_id": "LN1", "amount_due": 8500, "due_date": "2026-09-09", "days_overdue": 7, "lender": "Sunrise Finance"}


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path}/t.db")) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["db"] == "sqlite"


def test_respond_returns_tool_and_persists(client):
    r = client.post("/respond", json={"borrower_text": "Haan, main kal tak pay kar dunga", "account": ACCT})
    assert r.status_code == 200
    b = r.json()
    assert b["valid_json"] and b["tool"]["name"] and b["id"]
    assert client.app.state.core.store.count("responses") == 1


def test_respond_validation(client):
    bad = dict(ACCT, amount_due=0)
    assert client.post("/respond", json={"borrower_text": "hi", "account": bad}).status_code == 422
    assert client.post("/respond", json={"borrower_text": "", "account": ACCT}).status_code == 422
    r = client.post("/respond", json={"borrower_text": "hi", "account": ACCT, "variant": "nope"})
    assert r.status_code == 422


def test_evaluate_and_metrics(client):
    r = client.post("/evaluate", json={"variant": "pruned", "limit": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["n"] == 10 and "decision" in body["summary"]["pass_rate"]
    assert client.app.state.core.store.count("run_items") == 10
    m = client.get("/metrics").json()
    assert m["pruned"]["is_mock"] is True and m["pruned"]["run_id"] == body["run_id"]
    assert len(client.get("/runs").json()) == 1


def test_metrics_empty_before_any_run(client):
    assert client.get("/metrics").json() == {}
