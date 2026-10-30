"""Kernel source stays shape independent; dimensions belong in config memory."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'atalla-graph' / 'kernels'
PACKAGE = '_kernel_source_tests'
spec = importlib.util.spec_from_file_location(PACKAGE,ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = package
spec.loader.exec_module(package)


@pytest.mark.parametrize('name,first,second', [
    ('add',(7,),(2048,)), ('mul',(7,),(2048,)), ('relu',(7,7),(2048,32)),
    ('gemm',(1,7,4),(32,32,64)), ('softmax',(7,),(64,)),
    ('softmax_batched',(1,7),(3,32)), ('layernorm',(1,32,1e-5),(3,64,1e-6)),
    ('maxpool',(5,8,2,2,1),(5,32,2,3,1)),
])
def test_source_does_not_change_with_shape(name, first, second):
    fn = getattr(package, 'softmax_c_batched' if name == 'softmax_batched' else name+'_c')
    filename = 'softmax.c' if name == 'softmax_batched' else name+'.c'
    assert fn(*first) == fn(*second) == (ROOT/filename).read_text()


def test_generator_reads_external_source(monkeypatch):
    path = ROOT/'add.c'
    original = Path.read_text
    def read_text(self,*args,**kwargs):
        text = original(self,*args,**kwargs)
        return text.replace('int main()', 'int extracted_main()') if self == path else text
    monkeypatch.setattr(Path,'read_text',read_text)
    assert 'int extracted_main()' in package.add_c(32)
