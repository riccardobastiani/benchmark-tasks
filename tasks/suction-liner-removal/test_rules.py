"""Process-rule checks, not simulated robot or physics evidence."""
import importlib.util
from pathlib import Path
import sys

folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location('_suction_rules',folder/'task.py',
                                             submodule_search_locations=[str(folder)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
failure = module.process_failure
assert not failure(False,False,False,False,False)  # Correct untouched start.
assert not failure(True,True,True,True,False)      # Closed before isolated extraction.
assert failure(False,True,False,False,False)      # OFF with exposed port.
assert failure(True,False,True,False,False)       # Extraction while ON.
assert failure(False,False,False,True,False)      # Restoring ON cannot erase prior isolation.
assert failure(False,False,False,False,True)      # Lowering cannot erase prior extraction.
assert failure(False,False,True,False,False)      # Open extraction before any isolation.
print('Seven process-rule cases passed; native mechanics are checked separately.')
