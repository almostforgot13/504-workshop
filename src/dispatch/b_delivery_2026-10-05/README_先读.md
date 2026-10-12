# B 原计划任务：技术交付与四策略联调

打开整个文件夹，在 PyCharm 选择 Python 3.11，终端运行 `python -m pip install -r requirements.txt`，然后运行 `run_all_in_pycharm.py`。它先做测试，再重跑四策略，最后独立校验。新结果保存到 `results/run_时间戳/`，不覆盖附带的参考结果。四组均以同一 UTC 48 小时、同一设备/初始库存运行；优化策略共用同一配置与求解器。

## 文件位置

- `code/device_model.py`、`candidate_parameters_15min.json`：共同设备模型和唯一候选参数；参数来源见 `docs/parameter_sources_15min.csv`。
- `code/rule_dispatch.py`：已修复骤降问题的固定规则，也是优化无可行解时的显式安全回退。
- `code/milp_model.py`：2017 训练期单日参考检查，已复用共同求解器和终端零功率处理；它使用全天已知输入，不是四策略实验之一。
- `code/rolling_dispatch.py`：共同 96 步 MILP、状态传递、预测流式读取和实测纠正。
- `code/reserve.py`：q10—q90 宽度映射为有上下界的库存能量缓冲，不足时记录软缺口。
- `code/dispatch.py`：固定规则、持续性、q50、分位数备用的统一运行入口。
- `code/experiment_config.json`：冻结的演示目标、备用映射和求解限制；价格与惩罚尚待团队确认。
- `code/validate_dispatch.py`：四策略逐行独立复算；`test_*.py` 为边界、防未来信息和失败回退测试。
- `data/`：A 的实测输入、48 小时预测片段及 96 行接口样例；没有重训 A 模型。
- `results/reference_48h/`：本次实际生成的四组 192 行 CSV、汇总、日志与独立检查。
- `docs/B_DISPATCH_REPORT.md`：可交给 C 整合的设备/优化/备用/安全处理报告章节。
- `docs/B_TASK_STATUS.md`：按原始任务编号列出完成证据及待确认项。

## 直接命令

```powershell
python code/dispatch.py --actual data/A_15min_virtual_2017_2019.csv --forecast data/A_48h_forecasts.csv --start 2019-06-15T00:00:00Z --steps 192 --strategy all --output-dir results/new_run
python code/validate_dispatch.py --actual data/A_15min_virtual_2017_2019.csv --results-dir results/new_run
```

要运行 2019 完整预测，将 A 的 `test_2019_forecasts.csv.gz` 放入 data 并替换 `--forecast`，使用 `--start 2019-01-01T00:00:00Z --steps 34945`。输入按窗口流式读取；完整运行求解次数很多，当前附带结果是 48 小时工程验证，**不是已完成的全年成绩**。

## 如何读结果

先看 `verification_report.json` 是否通过、四组是否各 192 行；再看 `comparison_metrics.csv` 的失负荷、购电、剩余功率、回退、备用缺口和终端库存。`constraint_check.csv` 中每步可追溯 SOC/氢状态和检查结论。`fallback_log.csv` 包含限时解、回退、实测纠正及备用缺口事件；记录限时解不等于执行了回退。

所有 `step_cost` 留空、`total_cost=null`，不能当成零成本。备用配置是库存能量缓冲，不保证即时备用功率或整日可靠性。求解器可在 2% 最优性差距容差内结束，也可能在 2 秒上限时保留已验证的可行解；具体 gap 在日志中。数值可因机器速度变化。该 2019 片段曾用于联调，不是独立封存测试；不得按它反复调参后宣称独立测试优越性。
