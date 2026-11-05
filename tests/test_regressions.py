"""回归测试：锁死上一轮诊断出的三个 P0 缺陷与全仓路径清理，防止改回去。"""
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _code_files():
    """只扫仓库里的可执行文件。

    原实现走 os.walk，把工作区里 .gitignore 明确忽略的本地诊断脚本
    （_dbg_one.py、_probe3.bat 之类）也算成违规；那些文件根本不进仓库，
    拿它们判定交付质量是误报。改用 git ls-files，语义是"这个提交里
    真正会被别人克隆到的文件"。没有 git 时退回 os.walk 并排除临时目录。
    """
    try:
        out = subprocess.run(["git", "ls-files", "*.py", "*.bat"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=20, check=True).stdout.splitlines()
        # 断言本身带有这些反例字符串，tests/ 必须排除，否则自指误报
        return [os.path.join(ROOT, rel) for rel in out
                if rel.endswith((".py", ".bat"))
                and not rel.replace("\\", "/").startswith("tests/")]
    except (OSError, subprocess.SubprocessError):
        out = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
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

    def test_fmerge_does_not_rebuild_timeline(self):
        """9d437f3 的 P0：第二次 pairs 把 JJ[m] 与 (sid,m) 错配，IC 虚高一倍。"""
        src = _read("fmerge.py")
        # 注释里会引用这个反例，所以只禁"代码行"，不能整文件禁字符串
        bad = [l for l in src.splitlines()
               if "pairs.extend" in l and "(int(JJ[m]), sid, m)" in l]
        self.assertEqual(bad, [], "fmerge 仍在用错配的第二次时间轴重建")
        self.assertIn("assert np.all(np.diff(JJ) > 0)", src)
        self.assertIn("时间轴与段索引长度不一致", src)

    def test_fraw_requires_explicit_meta(self):
        """meta() 与 meta_unique() 的下标不通用，混用会静默算错因子行。"""
        src = _read("fraw.py")
        self.assertIn("def meta_unique()", src)
        self.assertIn('raise ValueError("fraw.row 需要显式传 _M', src)
        self.assertIn('raise ValueError("fraw.rows 需要显式传 _M', src)
        self.assertIn('raise ValueError("fraw.matrix 需要显式传 _M', src)
        self.assertNotIn("_M = meta() if _M is None else _M", src)

    def test_downstream_scripts_thread_meta_unique(self):
        import re
        for name in ("mlfit2.py", "risk_portfolio.py"):
            src = _read(name)
            self.assertIn("fraw.meta_unique()", src)
            # 只检查真正调用 fraw.rows/row/matrix 的代码行，跳过文档字符串
            import ast
            tree = ast.parse(src)
            code_lines = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if (isinstance(f, ast.Attribute)
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "fraw"
                        and f.attr in ("rows", "row", "matrix")):
                    code_lines.append(ast.get_source_segment(src, node) or "")
            for call in code_lines:
                self.assertTrue(re.search(r",\s*(M|C\[\"M\"\])\s*\)", call),
                                f"{name} 的 {call.strip()} 没传去重时间轴")

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
