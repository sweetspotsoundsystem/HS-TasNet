from pathlib import Path
import subprocess
import sys


def test_fixture_directories_have_no_transient_symlinks(tmp_path):
    fixture = tmp_path / "test_fixture.py"
    fixture.write_text('''
import os
def test_real_fixture_without_current_link(tmp_path):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['OMP_NUM_THREADS'] == '1'
    (tmp_path / 'actual-file.txt').write_text('count this real file')
    assert not any(p.is_symlink() for p in tmp_path.parent.rglob('*'))
''')
    runner = Path(__file__).resolve().parents[1] / "scripts/run_cpu_tests.py"
    result = subprocess.run([sys.executable, str(runner), str(fixture), "-q",
                             "--basetemp", str(tmp_path / "inner-fixtures")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert list((tmp_path / "inner-fixtures").rglob("actual-file.txt"))
