import subprocess
import sys


def test_controlled_benchmark_runs_and_reports_metrics():
    result = subprocess.run(
        [sys.executable, "-m", "app.evaluation.benchmark"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Controlled Retrieval Benchmark" in result.stdout
    assert "Recall@5" in result.stdout
    assert "MRR@5" in result.stdout
    assert "0.900" in result.stdout
    assert "1.000" in result.stdout
    assert "Per-query results" in result.stdout
