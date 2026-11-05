"""测试包：保证从仓库根目录执行 `python -m unittest discover -s tests -v` 可用。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
