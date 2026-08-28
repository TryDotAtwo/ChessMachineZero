from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "site" / "app.js").read_text(encoding="utf-8")


def test_site_describes_the_current_executable_artifact_exactly():
    for exact_contract in (
        "[1 × 2048 × 128]",
        "[1 × 3 × 128]",
        "[1 × 2045 × 128]",
        "[1 × 64 × 128]",
        "Generic ops × 45",
        "2064 events",
        "host chess logic",
        "none",
    ):
        assert exact_contract in HTML


def test_site_does_not_claim_unimplemented_recurrent_stages():
    assert "LEGAL_SET</code> and terminal status remain the next artifact stages" in HTML
    assert "Browser view replays an exact fixture" in HTML
    assert "policy head" not in HTML.lower()
    assert "tree search" not in HTML.lower()


def test_numeric_history_fixture_uses_three_native_tokens_per_move():
    assert "move: [52, 54, 96]" in JS
    assert "move: [57, 55, 102]" in JS
    assert "move: [71, 63, 97]" in JS
    assert "400 plies" in JS


def test_pages_deployment_publishes_only_the_static_site():
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "path: site" in workflow
    assert "actions/deploy-pages@v4" in workflow
