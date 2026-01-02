import tempfile
from pathlib import Path
import subprocess
import json


def test_generate_cluster_quick():
    tmp = Path(tempfile.mkdtemp(prefix='cluster_test_'))
    # Run generator with 5 subjects (fast)
    cmd = ['python', str(Path(__file__).parent / 'generate_cluster.py'), '--n-subjects', '5', '--out-dir', str(tmp), '--seed', '123']
    subprocess.check_call(cmd)

    # Check outputs
    assert (tmp / 'cluster_summary.json').exists()

    summary = json.loads((tmp / 'cluster_summary.json').read_text(encoding='utf-8'))
    assert summary['n_subjects'] == 5
    assert (tmp / 'subject_1_spec.json').exists()
    assert (tmp / 'subject_1.csv').exists()
    # combined CSV exists
    assert (tmp / 'combined_results.csv').exists()
