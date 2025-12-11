**20251210 PyTest Closed-Stream 问题交接**

- **目标**: 将仓库（非 tests）输出统一为 `loguru` 且在替换后保证 `pytest` 全部通过。

- **当前阻塞问题（优先级最高）**: pytest 在 session teardown/cleanup 阶段抛出
  `ValueError: I/O operation on closed file`，堆栈指向 `pytest` terminal writer ->
  `colorama.ansitowin32` -> `self.wrapped.write`。该异常使 pytest 以非零退出码失败。

- **已做排查与缓解（要点）**:
  - 在仓库根 `conftest.py` 加入探测器 `_StreamCloseDetector`，在导入时即包装 `sys.stdout`/`sys.stderr`，目的是记录第一次 `.close()` 的调用堆栈并替换为持久 `devnull`。
  - 补丁 colorama: patch `AnsiToWin32.__init__` 与 `AnsiToWin32.write`，对其内部的 `.wrapped` 和 `write` 做安全包装，捕获 `ValueError` 并把堆栈写到 repo `debug/closed-stream-tracebacks.log` 与 `sys.__stderr__`。
  - 对 pytest terminal writer 的 `_file` / `file` / `.wrapped.write` 做直接写封装（捕获 ValueError 即刻记录堆栈到 repo debug 并回显到 `sys.__stderr__`）。
  - 增加 import-time 标记 `debug/conftest-loaded.txt` 以确认 conftest 已被加载。

- **已运行的测试与观察**:
  - 在用户 pixi 环境下多次运行 `pytest -k "not test_manual_pure_numeric and not test_manual" -s`。
  - 大多数单元和集成测试可运行且产出预期分析文件（例如 warmup 导出的 `phase1_analysis_report.md` 等）。
  - 尽管加入了探测器与保护，pytest 最终仍在 teardown 抛 `ValueError`；conftest 的探测器被更新为在第一次 close() 时立即把堆栈打印到 `sys.__stderr__`，但仍未在 repo `debug/closed-stream-tracebacks.log` 见到调用者的 trace 文件（同时 stderr 回显也未捕到可供定位的闭流调用点）。

- **关键文件与位置**:
  - `conftest.py` (repo 根): 包含 `_StreamCloseDetector`、colorama 补丁、terminal writer 包装逻辑，写入 `debug/conftest-loaded.txt`。
  - 期望的探测日志: `debug/closed-stream-tracebacks.log`（若无则检查系统临时目录）。
  - 影响堆栈常见链: `pytest/_io/terminalwriter.py` -> `colorama/ansitowin32.py` -> `.wrapped.write`。
  - 其他相关修改: `extensions/dynamic_eur_acquisition/modules/diagnostics.py`（DiagnosticsManager 已重构为使用 loguru），warmup 样本 CSV 在 `extensions/warmup_budget_check/tests/sample/.../combined_results.csv`（已加入 `f1` 列以避免 sklearn 无特征错误）。

- **已知可疑调用模式（供快速审查）**:
  - 仓库中广泛存在 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)` 的重绑定模式（在 `tests/is_EUR_work` 等多处），这些会替换全局 stdout 对象，但没有直接 `close()` 的显式调用记录。
  - 搜索到的 `.close()` 调用多为数据库连接或 matplotlib 图像关闭（`conn.close()`、`plt.close()`），这些通常不会直接导致 pytest 写入终端时失败，但值得注意任何代码在退出时可能显式或隐式关闭被包装对象的 `.buffer`/底层文件句柄。

- **下一步建议（可直接执行）**:

 1. 复现并捕获第一次 close() 的完整调用堆栈（最可靠）：
    - 在本地运行完整 pytest（含 -s），观察 `sys.__stderr__` 中探测器回显的堆栈；或检查 `debug/closed-stream-tracebacks.log` 与系统临时目录中的同名日志。
    - 命令示例:

```
D:/ENVS/active-psych-sampling/.pixi/envs/default/python.exe -m pytest -q -s
```

 2. 若探测器仍未产出闭流堆栈：将 `conftest._StreamCloseDetector.close` 临时改为在检测到第一次 close() 时抛出一个自定义 Exception（会中断 pytest 但能把堆栈直接暴露），在定位到 offending module 后再移除该抛错逻辑。
 3. 审查并修复发现的 offending 代码：
    - 若是某模块/测试在关闭全局 `sys.stdout`/`sys.stderr`，将其改为只关闭局部打开的句柄，或用不关闭的包装器（non-closing proxy）。
    - 对第三方库（若为 culprit）采用包装器拦截其 close 调用。
 4. 完成 root cause 修复后：移除 conftest 的探测/掩盖逻辑（保持 conftest 清洁），继续仓库中非测试文件的 print/stderr -> loguru 替换。

- **接手人快速 checklist**:
  - 检查 `debug/closed-stream-tracebacks.log` 和 `%TEMP%/closed-stream-tracebacks.log`。
  - 在复现失败时把完整 pytest 输出（包含 conftest 探测回显）保存并粘贴给接手人。
  - 定位抛出 close() 的模块/测试（依据堆栈），修改后 rerun pytest 直到 exit code 0。

-- 文档作者: 自动化调试代理（生成时间 2025-12-10）
