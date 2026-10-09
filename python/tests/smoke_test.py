"""Import smoke test, run by CTest with PYTHONPATH pointing at the build tree.

Plain asserts on purpose: it must run with nothing but the interpreter.
"""

import turbocore

info = turbocore.build_info()
support = turbocore.runtime_support()

assert turbocore.__version__ == info["version"], (turbocore.__version__, info)
assert isinstance(info["avx2_compiled"], bool)
assert isinstance(info["cuda_compiled"], bool)
assert info["avx2_compiled"] or not support["avx2_usable"]
assert support["cuda_device_count"] >= 0

print(f"turbocore {turbocore.__version__} build={info} runtime={support}")
