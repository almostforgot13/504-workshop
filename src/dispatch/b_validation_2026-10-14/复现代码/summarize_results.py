from pathlib import Path
import csv,json,statistics,hashlib

OUT=Path(__file__).resolve().parent.parent;R=OUT/'结果'
METHODS=['A_rule','B_lgbm_q50','C_tcn_blend_q50','D_raw_reserve','E_cqr_reserve']
LABELS=['规则候选动作＋共同修正','LightGBM q50','TCN混合 q50','原始区间备用','校准区间备用']
def load(name):
    with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(name):return json.loads((R/name).read_text(encoding='utf-8'))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
new=load('新日期回放汇总.csv');hist=read('既有28天_独立审计.json');small=read('小样例_枚举参考解.json')
new_audits=[json.loads(p.read_text(encoding='utf-8')) for p in R.rglob('independent_audit.json')]
assert len(new_audits)==37 and all(a['passed'] for a in hist+new_audits)
assert all(x['passed'] for x in small)
totals=[]
for method,label in zip(METHODS,LABELS):
    rr=[r for r in new if r['strategy']==method]
    totals.append({'strategy':method,'label':label,'steps':sum(int(r['steps']) for r in rr),
        **{k:sum(float(r[k]) for r in rr) for k in ['operating_cost_cny','replacement_accounting_cny','adjusted_cost_cny','unserved_kwh','fallback_steps','load_kwh']}})
with (R/'四个新日期_按策略合计.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(totals[0]));w.writeheader();w.writerows(totals)
budget=[];points=load('求解预算_固定状态.csv')
for seconds in [.25,2.,10.,60.]:
    rr=[r for r in points if float(r['budget_s'])==seconds];g=[float(r['mip_gap'])*100 for r in rr]
    budget.append({'budget_s':seconds,'incumbents':len(g),'met_stop_criterion':sum(r['solver_status']=='optimal' for r in rr),'median_gap_pct':statistics.median(g),'max_gap_pct':max(g)})
comparison=load('整日预算对照.csv')
comparison=[r for r in comparison if r['start'].startswith('2019-01-14')]
nov25=json.loads((R/'新日期回放/20191111/C_tcn_blend_q50/summary.json').read_text(encoding='utf-8'))
nov2=json.loads((R/'强化压力测试/normal/C_tcn_blend_q50/summary.json').read_text(encoding='utf-8'))
comparison.extend([{**nov25,'comparison_budget_s':.25},{**nov2,'comparison_budget_s':2.}])
keys=list(dict.fromkeys(k for r in comparison for k in r))
with (R/'整日预算对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(comparison)
stress=load('强化压力测试汇总.csv');bounds=read('能量守恒_缺供下界.json');oracle=read('离线成本下界/求解证据.json')
warm=read('高功率回退_独立审计.json')
scenarios=['normal','renewable_drop_60pct','renewable_zero','forced_failure'];labels=['正常','风光持续下降60%','风光降为零','强制求解失败']
stress_table=[]
for scenario,label in zip(scenarios,labels):
    a=next(r for r in stress if r['scenario']==scenario and r['strategy']=='C_tcn_blend_q50')
    b=next(r for r in stress if r['scenario']==scenario and r['strategy']=='D_raw_reserve')
    cert=next(r for r in bounds if r['file'].replace('\\','/')==f'强化压力测试/{scenario}/C_tcn_blend_q50/dispatch.csv')
    stress_table.append([label,f"{float(a['unserved_kwh']):,.2f}",f"{float(b['unserved_kwh']):,.2f}",f"{cert['unserved_energy_lower_bound_kwh']:,.2f}"])
summary={'historical_trajectories':len(hist),'historical_steps':sum(x['steps'] for x in hist),'historical_checks':sum(x['checks'] for x in hist),
    'new_trajectories':len(new_audits),'new_steps':sum(x['steps'] for x in new_audits),'all_independent_audits_passed':True,
    'small_cases_passed':len(small),'small_reference_max_objective_difference':max(x['absolute_difference'] for x in small),
    'new_day_aggregate':totals,'solver_budget_certificates':budget,'warm_braking_audit':warm,'oracle':oracle,
    'max_physics_balance_residual_kw':max(x['max_errors']['balance_kw'] for x in hist+new_audits)}
(OUT/'实验结论.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
new_table=table(['策略','运行费（元）','末端补库记账（元）','合计（元）','失负荷（kWh）'],
    [[x['label'],f"{x['operating_cost_cny']:,.2f}",f"{x['replacement_accounting_cny']:,.2f}",f"{x['adjusted_cost_cny']:,.2f}",f"{x['unserved_kwh']:.2f}"] for x in totals])
budget_table=table(['预算','有可行解','达到停止准则','间隙中位数','最大间隙'],[[f"{x['budget_s']:g}秒",f"{x['incumbents']}/12",f"{x['met_stop_criterion']}/12",f"{x['median_gap_pct']:.4f}%",f"{x['max_gap_pct']:.4f}%"] for x in budget])
small_table=table(['样例','枚举参考费用','MILP费用','绝对差异（元）'],[[x['case'],f"{x['reference_objective']:.8f}",f"{x['milp_objective']:.8f}",f"{x['absolute_difference']:.3e}"] for x in small])
replay_table=table(['样例日','0.25秒费用','2秒费用','变化'],[[day,
    f"{float(next(r for r in comparison if r['start'].startswith(day) and float(r['comparison_budget_s'])==.25)['adjusted_cost_cny']):,.2f}",
    f"{float(next(r for r in comparison if r['start'].startswith(day) and float(r['comparison_budget_s'])==2.)['adjusted_cost_cny']):,.2f}",
    f"{(float(next(r for r in comparison if r['start'].startswith(day) and float(r['comparison_budget_s'])==2.)['adjusted_cost_cny'])/float(next(r for r in comparison if r['start'].startswith(day) and float(r['comparison_budget_s'])==.25)['adjusted_cost_cny'])-1)*100:+.2f}%"] for day in ['2019-01-14','2019-11-11']])
report=f'''# B 模块补充验证实验报告

日期：2026-10-14。使用最新初赛演示版的原始调度源代码、冻结预测模型及候选设备参数。未修改演示版 EXE、未重新训练模型、未根据新结果调参数。数据为公开曲线缩放的虚拟园区，费用为示例场景假设。

## 一、结果摘要

1. **基础物理计算通过验证。**独立审计既有20条轨迹、13,440步、376,360项检查；没有发现违规。新增37条整日轨迹、3,552步也全部通过，另外6步高功率回退通过。
2. **小规模优化结果可核对。**5个三步样例，每例枚举216种设备模式组合，参考费用与原MILP一致；最大差异约6.68×10^-10元。
3. **可以证明部分场景的最小缺供电量。**强化压力测试中，风光下降60%和降为零时，结果达到能量守恒下界。此结论针对当前模型的总缺供电量，不针对费用最优。
4. **求解预算影响单次规划质量。**12个固定状态下，0.25秒只有3次达到停止准则；2秒全部达到。10秒和60秒无进一步改善这些状态。
5. **备用优势尚未证实。**本轮压力测试中，q50与备用策略的总缺供相同。新日期中TCN q50合计费用最低，但仅4天，不代表全年或跨企业结论。
6. **经济最优性仍有提升空间。**诊断日的完全信息松弛参考费用约236.75元，在线信息条件下的q50回放约579.93元；不能把差额全部归因于预测，也不能承诺可实现同等节费。

## 二、实验范围与公平条件

- 原模型：15分钟步长、最大96步规划、末端库存欠额记账、共同单时段实测修正、原有安全回退。尾部窗口缩短仍保留。
- 既有审计：2019-01-14、04-15、07-15、10-14起始的四周，5种策略各672步。
- 新日期：2019-02-11、05-13、08-12、11-11，每日96步、5种策略、默认0.25秒预算。日期在新结果产生前固定，均在原四周基准之外；不保证此前从未在其他分析中查看过这些数据。
- 每次回放独立重置为同一初始库存；不是连续4天、连续全年运行。
- 求解预算对照：2019-01-14每2小时取一个既有TCN轨迹状态；每次使用相同完整96步预测、相同状态和费用，预算0.25/2/10/60秒，单线程。
- 首轮压力日2019-01-14全部零缺供。为扩大覆盖，保留首轮结果后，另固定11月11日正常、风光下降60%、风光为零和强制失败场景。这是诊断性扩展，不作为未经查看的独立留出测试。
- 压力测试保留原预测、缩减实际风光，模拟持续预测偏差。没有按变化后的观测重新生成预测，因此不是完整在线预测自适应实验。
- “规则”完整流程包含共同MILP修正；五个实验代号A–E不对应团队成员分工。

## 三、独立物理审计与检查器验证

检查器 `independent_check.py` 不导入原设备检查函数，直接从执行CSV重算功率平衡、状态递推、额定功率、最小开机功率、互斥、爬坡、库存、未来停机库存、用户缺供上限、电价与费用，并对照声明的原始实测数据。

既有和新增整日轨迹最大功率平衡残差：{summary['max_physics_balance_residual_kw']:.3e} kW。检查容差1×10^-5。人为注入功率、库存、费用、负功率和重复时间等6类错误，全部被识别。

独立实现能减少共同代码错误，但仍建立在同一简化物理假设上，不能代替真实设备验证。极小残差也不能代表零缺供。

## 四、小样例参考解

{small_table}

每例3个时段；每步电池两种模式×氢设备三种模式，共6^3=216种组合。固定模式后独立建立连续线性规划，再比较最优目标值。允许存在多个同费用方案，不要求动作逐项相同。参考LP与MILP均使用HiGHS，但模型独立构造，模式通过穷举处理；不是不同求解器交叉验证。

## 五、四个新日期策略对照

{new_table}

TCN q50在本组样例中合计最低。规则实际运行费用较低，却大量消耗初始库存，补库记账使合计费用较高。因此不能把合计差异直接解释为实际电费节省。校准备用的1.42×10^-14 kWh属于浮点舍入，显示为0。

先前28天中LightGBM表现更好，与本组排名不同。结果支持“效果依赖场景”，不支持“TCN或备用始终最好”。

## 六、求解预算与整日效果

{budget_table}

“达到停止准则”对应求解器成功状态，仍按配置1%间隙目标停止；不是要求零间隙。有些限时结果可能已经很接近目标，状态与最终间隙都应报告。

同预测同参数的整日TCN q50回放：

{replay_table}

1月日略有改善，11月日反而增加。求解更充分可以改善当前预测问题的优化质量，但未来预测并不准确，滚动轨迹及末端记账会变化，因此最终实际费用不保证单调改善。墙钟预算结果也受硬件和系统负载影响。

## 七、压力测试与缺供下界

11月11日，两策略均2秒预算：

{table(['情景','q50缺供（kWh）','备用缺供（kWh）','能量下界（kWh）'],stress_table)}

### 不需要参考调度，也能推导下界

设全天总负荷为L，风光总电量为R，购电限额为Gmax，时长T；初始电池、氢库存分别为E0、H0，最低库存为Emin、Hmin。

    最低必要缺供 >= max(0, L - R - Gmax×T
                          - η_discharge×(E0-Emin)
                          - 燃料电池kWh/kg×(H0-Hmin))

推导依据：对全时段功率平衡求和，再代入库存递推。充放电往返效率≤1，制氢再发电效率≤1；弃电、转换损失均不能增加可用能源。因此负荷中超过“全部风光＋最多购电＋初始可用库存”的部分必然缺供。

11月11日总负荷26,682.66kWh；最多购电12,000kWh；初始电池可供电645.37kWh；初始氢可发电4,290kWh。

- 风光下降60%：风光6,456.05kWh，下界3,291.24kWh；两策略均达到下界。
- 风光降为零：下界9,747.29kWh；两策略均达到下界。
- 下界与执行结果差约4.6×10^-8kWh，处于数值容差内；结合物理可行性，可证明这两个情景的**总缺供电量在模型内已达最小值**。

这不证明缺供时刻分配唯一、成本最优或真实系统性能。下界为零也不自动证明某策略能零缺供。备用没有减少这两个场景的总缺供，因为既有能源条件已经限制了总供电能力。

### 求解失败与高功率回退

整日强制失败时两策略均96步回退，缺供830.48kWh，仍通过物理检查。它说明安全回退可运行，不说明回退具有良好经济性。

高功率合成案例从电池550kWh、氢75kg、上一步电池放电/燃料电池均500kW开始；两设备依次降为375、250、125、0、0、0kW。最终电池{warm['terminal_battery_kwh']:.2f}kWh、氢{warm['terminal_hydrogen_kg']:.2f}kg，保持在下限以上。设备停下后仍缺供，安全不等于保供。降功率期间的总线余电可能来自燃料电池，不应统一标为弃风弃光。

## 八、完全信息离线成本参考

独立建立1月14日96步参考模型，输入整天实际净负荷，使用相同电价、额定功率、互斥、爬坡、实际客户负荷上限、启动费用与末端库存欠额费用；放宽未来停机库存要求和末段功率归零要求。

- 参考最优费用/求解器下界：{oracle['objective_cny']:.8f}元；间隙0；实际求解约{oracle['wall_seconds']:.3f}秒，300秒为上限。
- q50实际回放费用：579.93138499元；备用：628.85486809元。
- 将两条实际执行轨迹代入参考模型，约束均满足容差，证明该参考确实包含这两条轨迹，费用口径一致。
- 参考解另行通过普通物理和停机库存检查，但它需要未来实际信息，且参考未要求末段功率归零，仍不能当成待部署算法。

所以236.75元是一个乐观参考下界，差距同时涉及未来信息、约束放宽和滚动执行；不能全部归因于预测误差，也不能保证在线能达到该费用。

## 九、向评委可陈述的结论

“我们通过独立物理审计、可枚举的小规模参考解、求解器上下界、统一条件策略对照以及故障压力测试验证调度。部分缺能场景还能用能量守恒证明最低缺供量，并确认执行结果达到该下界。当前证据支持离线调度原型的物理一致性和部分场景的最小缺供能力；备用的额外收益、全年经济性及真实在线控制仍需进一步验证。”

不能声称已证明全年最优、所有场景不断电、真实企业节费或设备可直接部署。建议下一阶段研究持续48小时以上运行、固定96步窗口、独立留出长时段及真实设备参数，明确实测平均值修正与在线观测的区别。
'''
(OUT/'实验结果报告.md').write_text(report,encoding='utf-8')
readme='''# 实验包阅读与复现

先看 `实验结果报告.md` 和 `图像/07_最低缺供的数学证据.png`。图片共7张，Python生成、宋体、无PDF。`结果/`保存汇总CSV、逐步轨迹、规划日志、求解器证据和独立审计。`输入/`含冻结模型、候选参数、原始虚拟数据和既有轨迹；不含API密钥。`复现代码/源程序/`是原调度与预测源码快照。

复现使用Python3.12及运行环境JSON记录的版本。实验运行时使用工作区中原演示版的嵌入Python；该解释器没有随结果包复制。已有环境可按顺序运行：

```powershell
$py = '你的Python3.12解释器绝对路径'
& $py '复现代码/run_experiments.py' checks
& $py '复现代码/run_experiments.py' forecasts
& $py '复现代码/run_experiments.py' new_days
& $py '复现代码/run_experiments.py' stress
& $py '复现代码/run_experiments.py' snapshots
& $py '复现代码/run_experiments.py' stress_extra
& $py '复现代码/offline_reference.py'
& $py '复现代码/energy_certificates.py'
& $py '复现代码/summarize_results.py'
& $py '复现代码/render_figures.py'
```

运行前将终端工作目录切到本包根目录。数值依赖：NumPy、Pandas、SciPy、PyTorch CPU、LightGBM；绘图依赖Pillow和Windows宋体。版本见 `运行环境.json`。非Windows绘图时需要修改宋体路径。代码与模型在不同CPU架构下可能出现微小浮点差异。

整日任务检测到既有 `summary.json` 时会复用结果，固定状态预算检测到相同小时和预算也会复用。要重新计算，请将整个 `结果/`目录重命名备份，再从checks阶段顺序运行；预测重新生成但冻结模型不训练。不要将旧结果与新环境结果混合比较。原数据和既有轨迹已经随包提供，复现不依赖网络。

LP枚举与MILP均统一使用单线程，避免同一进程的HiGHS线程调度设置冲突。实验只给原函数增加求解结果记录，不改变原调度约束。离线成本参考及能量下界是独立验证模型，并非更新后的产品算法。
'''
(OUT/'阅读与复现说明.md').write_text(readme,encoding='utf-8')
print(json.dumps({'trajectories':len(new_audits),'new_steps':summary['new_steps'],'summary_report_created':True,'new_days_costs':totals},ensure_ascii=False))
