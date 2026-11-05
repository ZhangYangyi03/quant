"""回归测试：锁死上一轮诊断出的三个 P0 缺陷与全仓路径清理，防止改回去。"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _code_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # tests/ 里的断言本身就包含这些反例字符串，扫描时排除，避免自指误报
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", ".workbuddy-ai", "tests")]
        for fn in filenames:
            if fn.endswith((".py", ".bat")):
                out.append(os.path.join(dirpath, fn))
    return out


class TestP0Fixes(unittest.TestCase):
    def test_walkforward_now_sorts_by_ep(self):
        src = _read("walkforward.py")
        self.assertIn("cand.sort(key=lambda x: -x[1])", src,
                      "缺失 EP 排序会让分年收益测的不是策略")
        self.assertIn("不是参数 walk-forward", src)

    def test_equal_weight_benchmark_is_now_deterministically_sampled(self):
        src = _read("backtest_r7.py")
        self.assertIn('elif which.startswith("eq_"):', src)
        self.assertIn("hashlib.sha256", src)
        self.assertIn("d[:7]", src, "抽样必须逐调仓月重新洗牌")
        self.assertIn("2026-11-05 修复", src)

    def test_readme_marks_invalid_numbers(self):
        src = _read("README.md")
        for token in ("机构级改造说明", "24.14%", "17.45pp", "11.19%",
                      "quantlab/", "INSTITUTIONAL.md"):
            self.assertIn(token, src, f"README 缺少 {token}")

    def test_no_hardcoded_old_machine_paths_in_executable_files(self):
        offenders = []
        for path in _code_files():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            for bad in (r"D:\Users\china", r"C:\Users\china",
                        r"miniconda3\python.exe"):
                if bad.lower() in text.lower():
                    offenders.append((os.path.relpath(path, ROOT), bad))
        self.assertEqual(offenders, [], f"仍有硬编码旧机器路径: {offenders}")

    def test_bat_scripts_use_portable_python(self):
        for name in ("_fund.bat", "_run.bat", "_tick.bat"):
            src = _read(name)
            self.assertIn("%~dp0", src)
            self.assertNotIn("miniconda", src.lower())


class TestPackageIntegrity(unittest.TestCase):
    def test_all_quantlab_modules_compile(self):
        bad = []
        base = os.path.join(ROOT, "quantlab")
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                with open(p, "r", encoding="utf-8") as f:
                    src = f.read()
                try:
                    compile(src, p, "exec")
                except SyntaxError as exc:
                    bad.append((os.path.relpath(p, ROOT), str(exc)))
        self.assertEqual(bad, [], f"语法错误: {bad}")

    def test_expected_deliverables_exist(self):
        for name in ("INSTITUTIONAL.md", "run_institutional.py",
                     os.path.join("quantlab", "validate", "no_lookahead.py"),
                     os.path.join("quantlab", "reporting.py")):
            self.assertTrue(os.path.exists(os.path.join(ROOT, name)),
                            f"缺少交付物 {name}")

    def test_version_is_declared(self):
        import quantlab
        self.assertEqual(quantlab.__version__, "1.0.0")


if __name__ == "__main__":
    unittest.main()
