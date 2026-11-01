"""Exercise address-budget boundaries without allocating large buffers."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen import layout_emitter
from graph.layout import LayoutOp, TensorView


@pytest.mark.parametrize('limit,message', [(4099, 'buffers'), (4107, 'snapshot')])
def test_address_budget_rejects_buffers_or_snapshot(monkeypatch, limit, message):
    monkeypatch.setattr(layout_emitter, 'MAX_DATA_ADDRESS', limit)
    operation = LayoutOp('copy', TensorView('a', (1,), 'i32'), TensorView('b', (1,), 'i32'))
    with pytest.raises(OverflowError, match=message):
        layout_emitter.emit_layout(operation, {'a': bytes(4), 'b': bytes(4)})


def test_address_budget_accepts_exact_upper_boundary(monkeypatch):
    monkeypatch.setattr(layout_emitter, 'MAX_DATA_ADDRESS', 4108)
    operation = LayoutOp('copy', TensorView('a', (1,), 'i32'), TensorView('b', (1,), 'i32'))
    packed = layout_emitter.emit_layout(operation, {'a': bytes(4), 'b': bytes(4)})
    assert packed.buffer_addresses == {'a': 4096, 'b': 4100}
