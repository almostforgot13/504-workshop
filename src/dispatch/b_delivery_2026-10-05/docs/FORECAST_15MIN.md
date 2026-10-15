# 15分钟预测接口（v0.2，A预测交付）

本规范适用于当前A的15分钟预测和B的0.25 h设备模型。原小时级样例与clean/forecast/dispatch校验模式继续保留，只用于历史接口测试；不能用来验收本次96步输出。

## 清洗数据

data/processed/A_15min_virtual_2017_2019.csv，time为UTC区间起点，load_kw/pv_kw/wind_kw为区间平均功率；净负荷=load-pv-wind，可为负。其余来源字段、split和quality_flag保留。2017/2018/2019对应训练/验证/测试。

## 预测列与时间

| 列 | 定义 |
|---|---|
|issue_time|UTC预测发起时刻，只能用time < issue_time的已结束观测|
|target_time|被预测15分钟区间的起点|
|horizon_step|整数1—96|
|horizon_hours|horizon_step × 0.25，预测区间结束距发起时刻的小时数|
|q10_kw/q50_kw/q90_kw|净负荷分位数，有限数值，q10 ≤ q50 ≤ q90，可为负|

**target_time = issue_time + (horizon_step − 1) × 15分钟**。08:00发起时最后已知观测的time是07:45；第1点预测08:00—08:15，第96点是次日07:45—08:00。horizon_hours不是target_time − issue_time，不能沿用旧小时接口的公式。

同一发起时刻必须恰好96行，按提前量排序；发起时刻递增。输出只包含上述7列，不带未来真实负荷、风光或质量标记。连续发起时刻每15分钟一次。

## 调度和评估边界

Δt=0.25 h，kW×0.25才是每步kWh；氢量仍为kg，引用B的candidate_parameters_15min.json。q50为中位数中心预测，不是数学期望。区间只是边际预测范围，实测覆盖率72.30%，不能当作全天80%可靠性保证，也不能平均四个分位数便声称得到小时分位数。

2019仅评估完整96步均在2019内的34,945个发起点；年末不足24小时不发起新测试窗口。保持模型固定，历史观测逐步可用。测试主指标排除容量/出力率重建的目标点，同时报告含它们的指标。模型调优与备用参数选择不得使用测试期结果。

滚动优化每次规划96步、仅执行第1步，再用实测更新设备状态。B现有规则策略不读预测；本交付不更改B的设备约束、参数、策略命名或成本空值定义。

## 验收

```sh
python tools/validate_contract.py clean15 data/processed/A_15min_virtual_2017_2019.csv
python tools/validate_contract.py forecast15 dataprogram/example_96_steps.csv
python tools/validate_contract.py forecast15 dataprogram/test_2019_forecasts.csv.gz
python -m unittest discover -s tests -v
```

B调度结果继续用B的专用validate_results.py复算，旧小时dispatch模式不能验收B的15分钟输出。成本尚未定义时保持空值，不填0冒充。
