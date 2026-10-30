import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'atalla-graph/scripts/pipelining_latency_model.py'
spec = importlib.util.spec_from_file_location('latency_model', SCRIPT)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_manual_cli_uses_supplied_workload():
    result = cli('--name', 'probe', '--bytes-loaded', 128, '--bytes-written', 0,
                 '--compute-cycles', 20, '--tiles', 2, '--assume-packed-transfers', '--lat-scpad', 10)
    assert result.returncode == 0, result.stderr
    assert 'probe' in result.stdout and 'add_2048' not in result.stdout
    assert 'Unvalidated' in result.stdout


@pytest.mark.parametrize('loads,stores,compute,tiles,serial,pipelined', [
    (64, 64, 10, 1, 30, 30),
    (128, 128, 20, 2, 60, 50),
    (128, 0, 60, 2, 80, 70),
    (256, 0, 20, 2, 60, 50),
    (0, 0, 0, 2, 0, 0),
    # Two one-byte tiles each require a separate transaction.
    (2, 0, 20, 2, 40, 30),
])
def test_hand_calculated_uniform_tiles(loads, stores, compute, tiles, serial, pipelined):
    r = model.evaluate_pipelining('case', loads, stores, compute, tiles, lat_scpad=10)
    assert r.t_serial == serial
    assert r.t_pipelined == pipelined


def test_separate_transfers_round_independently_and_overhead_can_lose():
    r = model.evaluate_pipelining('case', 4, 0, 20, 2, lat_scpad=10,
                                  load_transfer_bytes=[1, 1], pipeline_overhead_cycles=30)
    assert r.dma_lines_load == 4
    assert r.t_serial == 60
    assert r.t_pipelined == 80
    assert r.projected_speedup < 1


@pytest.mark.parametrize('kwargs', [dict(tiles=0), dict(bytes_loaded=-1), dict(line_bytes=0),
                                    dict(lat_scpad=float('nan'))])
def test_invalid_model_inputs(kwargs):
    values = dict(name='bad', bytes_loaded=0, bytes_written=0, compute_packets=0)
    values.update(kwargs)
    with pytest.raises(ValueError):
        model.evaluate_pipelining(**values)


def test_perf_requires_explicit_compute_and_transfer_assumptions(tmp_path):
    perf = tmp_path / 'x_perf.out'
    perf.write_text('bytes_loaded: 128\nbytes_stored: 0\npackets_executed: 20\n')
    assert cli('--perf', perf, '--tiles', 2).returncode != 0
    result = cli('--perf', perf, '--tiles', 2, '--use-packets-as-compute-proxy', '--assume-packed-transfers')
    assert result.returncode == 0, result.stderr
    assert 'proxy' in result.stdout


@pytest.mark.parametrize('args', [[], ['--perf', '/nonexistent', '--tiles', '2'],
    ['--bytes-loaded', '64', '--bytes-written', '0', '--compute-cycles', '10'],
    ['--dir', '/nonexistent', '--tiles', '2']])
def test_incomplete_or_missing_inputs_fail(args):
    assert cli(*args).returncode != 0


def test_explicit_transfer_cli():
    result = cli('--name', 'separate', '--tiles', 2, '--load-transfer-bytes', 1,
                 '--load-transfer-bytes', 1, '--store-transfer-bytes', 0,
                 '--compute-cycles', 20, '--lat-scpad', 10)
    assert result.returncode == 0, result.stderr
    assert 'separate' in result.stdout and '1.200' in result.stdout


def test_directory_and_malformed_perf(tmp_path):
    perf = tmp_path / 'first_perf.out'
    perf.write_text('bytes_loaded: 128\nbytes_written: 0\npackets_executed: 20\n')
    args = ('--dir', tmp_path, '--tiles', 2, '--assume-packed-transfers',
            '--use-packets-as-compute-proxy')
    result = cli(*args)
    assert result.returncode == 0, result.stderr
    assert 'first' in result.stdout
    perf.write_text('bytes_loaded: nonsense\n')
    assert cli(*args).returncode != 0


def test_transfer_geometry_must_match_totals():
    with pytest.raises(ValueError):
        model.evaluate_pipelining('bad', 128, 0, 20, 2, load_transfer_bytes=[1])
    with pytest.raises(ValueError):
        model.evaluate_pipelining('bad', 3, 0, 20, 2)
