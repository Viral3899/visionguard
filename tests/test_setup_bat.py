from pathlib import Path

def test_setup_bat_structure():
    root = Path(__file__).resolve().parent.parent
    bat = root / 'setup.bat'
    assert bat.is_file(), "setup.bat must exist in project root"
    content = bat.read_text(encoding='utf-8', errors='ignore')
    assert ".venv" in content
    assert "setup_models.py" in content
    assert "requirements.txt" in content
    assert "run.bat" in content
