"""Public loader contract shared by CLI, scripts and graph capture."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_graph import load_model


@pytest.mark.parametrize('name,input_shape,output_shape', [
    ('basic', (1, 32), (1, 32)),
    ('alexnet_small', (1, 3, 32, 32), (1, 10)),
    ('layernorm_smoke', (1, 32), (1, 32)),
    ('vit_micro', (1, 32, 32), (1, 32, 32)),
])
def test_loader_returns_model_and_executable_input(name, input_shape, output_shape):
    model, x = load_model(name)
    assert tuple(x.shape) == input_shape
    with torch.no_grad():
        assert tuple(model(x).shape) == output_shape


def test_alexnet_scale_is_used():
    model, _ = load_model('alexnet_small', scale=0.125)
    assert model.conv1.out_channels == 8


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match='Unknown model'):
        load_model('unknown')


@pytest.mark.parametrize('name', ['basic', 'alexnet_small', 'layernorm_smoke', 'vit_micro'])
def test_cli_uses_forward_loader_without_training_dependencies(monkeypatch, tmp_path, name):
    import run_graph as runner
    captured = {}

    def capture(model, x, **kwargs):
        captured['capture_kwargs'] = kwargs
        return object()

    def validate(gm, model, x, out_dir, **kwargs):
        captured['validation_kwargs'] = kwargs
        return {'strict_pass': True}

    monkeypatch.setattr(sys, 'argv', ['run_graph.py', '--model', name, '--mode', 'validate',
                                     '--out-dir', str(tmp_path), '--quiet'])
    monkeypatch.setattr(runner, 'build_graph', capture)
    monkeypatch.setattr(runner, 'run_validate', validate)
    runner.main()
    assert 'example_target' not in captured['capture_kwargs']
    assert 'include_backward' not in captured['capture_kwargs']
    assert 'expected_output' not in captured['validation_kwargs']
    assert 'training' not in captured['validation_kwargs']


def test_kernel_cli_rejects_training_flag(monkeypatch):
    import run_graph as runner
    monkeypatch.setattr(sys, 'argv', ['run_graph.py', '--training'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
