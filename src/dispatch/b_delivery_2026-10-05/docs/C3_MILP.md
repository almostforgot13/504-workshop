# B / C3：15 分钟样例日 MILP

这是供团队审阅的 C3 小样本优化器。使用现有候选参数，不调整设备容量。输入为 A 的虚拟园区 15 分钟数据，示例取 2017-01-05 UTC，共 96 步。数据属于训练期，不能作为封存测试成绩。

## 运行

需要 Python 3.10+、NumPy 和 SciPy（使用 `scipy.optimize.milp` / HiGHS）。从仓库根目录运行：

```powershell
python src/dispatch/b_module_15min/milp_model.py --input data/processed/A_15min_virtual_2017_2019.csv --day 2017-01-05 --output results/c3_milp/sample_day_2017-01-05.csv
```

同时写出同名 `.json` 求解报告。无可行解或物理检查失败时不写 CSV，只保存失败报告并退出；有验证后的限时可行解可写 CSV，但保留状态 1 和 gap。必须使用新的输出文件名，避免旧结果冒充成功。样例位于 `src/dispatch/b_module_15min/sample_day_2017-01-05.csv` 及同名 `.json`。

样例 CSV 使用 B 模块约定的 18 列调度接口；`strategy=clairvoyant_training_day_reference`，状态为 `optimal` 或 `time_limit_feasible`。由于团队尚未确定成本口径，`step_cost` 留空，演示目标值只在 JSON 中报告。

## 约束与核验

- 15 分钟功率平衡；购电上限；失负荷不超过客户负荷；弃电/总线多余功率显式记录。
- 电池充放电互斥；制氢槽与燃料电池互斥；两者最小开机功率和最大功率。
- 电池能量及氢库存递推与上下限；电池有符号净功率、制氢槽、燃料电池逐步爬坡。
- 求解后逐步调用 `device_model.apply_dispatch` 独立于矩阵约束的设备状态检查，报告最大功率平衡残差。

目标权重为演示假设：购电 1、弃电 0.1、失负荷 100。**这些不是团队确认的电价或成本参数。** 当前已共用滚动求解器，窗口末端设备功率设为零，第一步保留制动库存；日末经济库存价值仍未确定，不能将此完全预知的训练日目标值用于四策略优劣比较。正式滚动入口见 `dispatch.py` 和 [B报告](B_DISPATCH_REPORT.md)。

## 本次样例结果

2017-01-05：HiGHS 在设定容差内完成，MIP gap 约 0.0522%；96 步物理检查通过，最大平衡残差约 `4.8e-12 kW`。演示目标值 `16777.39`，对应购电 `8788.41 kWh`、弃电 `0`、失负荷 `79.89 kWh`。日末电池 `300 kWh`、储氢约 `60.49 kg`，四个设备功率为零。保守末端零功率处理会影响该日供电，不能拿它与旧无终端约束原型直接比较成绩。

优化器直接读取 B 模块目录现有的参数和设备模型，复用 `rolling_dispatch.solve_window`。原始交付快照保存在 GitHub 的 C3 原型分支；当前本地版本同时含规则骤降修复、动态库存备用和显式回退。
