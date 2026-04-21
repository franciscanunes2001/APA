"""
Entry point — run from the project root:
    python run_agent.py
    python run_agent.py --max-iter 10 --target-f1 0.84
"""
from disaster_agent.agent import main

if __name__ == "__main__":
    main()
