from pathlib import Path


def find_latest_run(base_name: str, runs_dir: Path = Path("runs")) -> Path:
    candidates = sorted(runs_dir.glob(f"{base_name}*/"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1] / "checkpoints/best_agent.pt"
    return runs_dir / base_name / "checkpoints/best_agent.pt"
