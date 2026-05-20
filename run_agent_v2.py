"""
Entry point for the v2 agent.

Run from the project root in either of these ways:

    python -m disaster_agent_v2.run_agent_v2
    python -m disaster_agent_v2.run_agent_v2 --max-iter 10 --target-f1 0.84
    python disaster_agent_v2/run_agent_v2.py
"""

import sys
from pathlib import Path

# Allow running this file as a plain script (not just `python -m ...`).
# When executed directly, __package__ is empty and the relative import
# below would fail — so add the project root to sys.path first.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from disaster_agent_v2.agent_v2 import main
else:
    from .agent_v2 import main


if __name__ == "__main__":
    main()
