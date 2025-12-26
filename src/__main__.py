"""
OiaFed 包入口点

支持以下运行方式:
    python -m oiafed run --config config.yaml
    python -m oiafed validate --config config.yaml
    python -m oiafed list
    python -m oiafed version
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
