"""Static PNG evidence figures, drawn in Python with SimSun. No PDF."""
from pathlib import Path
import csv,json,math,statistics
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent.parent;R=OUT/'结果';FIG=OUT/'图像';FIG.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/simsun.ttc';INK='#17304a';MUTED='#596d7f';GRID='#dce5ec';BLUE='#2878ba';ORANGE='#dd9340';GREEN='#24877d';RED='#bd534f'
METHODS=['A_rule','B_lgbm_q50','C_tcn_blend_q50','D_raw_reserve','E_cqr_reserve']
LABELS=['规则＋修正','LightGBM q50','TCN q50','原始区间备用','校准区间备用']
COLORS=['#8b98a4','#7798bc',BLUE,GREEN,'#665aa3']
def load(name):
    with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def f(v):return float(v) if v not in (None,'','nan') else float('nan')
def font(size):return ImageFont.truetype(FONT,size,index=0)
def text(d,xy,s,size=28,color=INK,anchor=None):
    s=str(s).replace('−','-').replace('⁻⁵','^-5').replace('⁻⁴','^-4')
    d.text(xy,s,font=font(size),fill=color,anchor=anchor)
def canvas(title,subtitle,height=1380):
    im=Image.new('RGB',(1800,height),'white');d=ImageDraw.Draw(im)
    d.rectangle((66,55,77,109),fill=BLUE);text(d,(99,49),title,48);text(d,(68,127),subtitle,27,MUTED)
    return im,d
def foot(d,h,note):
    d.line((68,h-112,1732,h-112),fill=GRID,width=2)
    text(d,(68,h-96),note,24,MUTED);text(d,(68,h-57),'UTC · 15分钟步长 · 虚拟园区 · 参数和费用为候选场景假设',23,MUTED)
def axis(d,box,ymax,label,ticks=5):
    x0,y0,x1,y1=box
    for k in range(ticks+1):
        value=ymax*k/ticks;y=y1-(y1-y0)*k/ticks
        d.line((x0,y,x1,y),fill=GRID,width=1);text(d,(x0-14,y),f'{value:g}',24,MUTED,'rm')
    d.line((x0,y0,x0,y1),fill=MUTED,width=2);text(d,(x0,y0-41),label,25,MUTED)
    return lambda value:y1-(y1-y0)*value/ymax
def table(d,x,y,widths,headers,rows,rowh=65,highlight=None):
    xend=x+sum(widths);d.rectangle((x,y,xend,y+rowh),fill='#eaf1f7')
    for ri,row in enumerate([headers]+rows):
        yi=y+ri*rowh;xx=x
        if ri and ri%2==0:d.rectangle((x,yi,xend,yi+rowh),fill='#f7fafc')
        for ci,(value,w) in enumerate(zip(row,widths)):
            text(d,(xx+15,yi+rowh/2),value,27,BLUE if highlight and highlight(ri,ci) else INK,'lm');xx+=w
        d.line((x,yi+rowh,xend,yi+rowh),fill=GRID,width=1)
def save(im,name):im.save(FIG/name)

def proof():
    audits=json.loads((R/'既有28天_独立审计.json').read_text(encoding='utf-8'))
    small=json.loads((R/'小样例_枚举参考解.json').read_text(encoding='utf-8'))
    mutations=json.loads((R/'独立检查器_错误注入.json').read_text(encoding='utf-8'))
    im,d=canvas('基础正确性：独立检查与小样例参考','设备账本重算不调用原检查函数；小样例枚举开关组合后求连续功率。',1230)
    cards=[('既有轨迹',f"{sum(a['steps'] for a in audits):,} 时段",'20条轨迹均通过'),('错误注入',f"{sum(m['detected'] for m in mutations)}/{len(mutations)} 类被识别",'检查器能拒绝错误记录'),('枚举参考',f"{sum(s['passed'] for s in small)}/{len(small)} 样例一致",'每例216种开关组合')]
    for i,(title,value,note) in enumerate(cards):
        x=68+i*566;d.rounded_rectangle((x,205,x+535,400),radius=15,fill='#edf5f9')
        text(d,(x+25,227),title,29,MUTED);text(d,(x+25,274),value,41,BLUE);text(d,(x+25,344),note,27)
    labels=['风光富余','持续缺电','富余转缺电','库存接近下限','库存接近上限']
    rows=[[label,f"{s['reference_objective']:,.6f}",f"{s['milp_objective']:,.6f}",f"{s['absolute_difference']:.2e}"] for label,s in zip(labels,small)]
    table(d,68,475,[380,450,450,384],['三步样例','枚举最优费用（元）','原MILP费用（元）','差异（元）'],rows)
    maxres=max(a['max_errors']['balance_kw'] for a in audits)
    text(d,(68,925),f"累计 {sum(a['checks'] for a in audits):,} 项检查；最大功率平衡误差 {maxres:.2e} kW。",29)
    text(d,(68,986),'通过表示这些轨迹和小样例一致；不能替代真实设备或电压、频率验证。',28,RED)
    foot(d,1230,'检查容差：1×10⁻⁵；小样例参考费用一致容差：1×10⁻⁴元。');save(im,'01_基础正确性.png')

def new_days():
    rows=load('新日期回放汇总.csv');totals={m:sum(f(r['adjusted_cost_cny']) for r in rows if r['strategy']==m) for m in METHODS}
    im,d=canvas('新日期测试：四个样例日的费用与缺供','2月11日、5月13日、8月12日、11月11日；模型和设备参数未调优。',1550)
    box=(170,255,1715,665);ymax=math.ceil(max(totals.values())/10000)*10000;Y=axis(d,box,ymax,'情景比较成本（元）')
    for i,m in enumerate(METHODS):
        x=245+i*290;rr=[r for r in rows if r['strategy']==m];oper=sum(f(r['operating_cost_cny']) for r in rr);value=totals[m]
        d.rectangle((x,Y(value),x+145,665),fill=ORANGE);d.rectangle((x,Y(oper),x+145,665),fill=COLORS[i])
        text(d,(x+72,Y(value)-20),f'{value:,.0f}',29,INK,'mb');text(d,(x+72,690),LABELS[i],26,INK,'mt')
    text(d,(68,756),'柱内橙色：末端库存欠额记账；其他颜色：已执行动作费用。',26,MUTED)
    table_rows=[]
    for i,m in enumerate(METHODS):
        rr=[r for r in rows if r['strategy']==m]
        table_rows.append([LABELS[i],f"{sum(f(r['operating_cost_cny']) for r in rr):,.2f}",f"{sum(f(r['replacement_accounting_cny']) for r in rr):,.2f}",f"{totals[m]:,.2f}",f"{sum(f(r['unserved_kwh']) for r in rr):.2f}"])
    table(d,68,824,[365,330,330,330,309],['策略','运行费用（元）','补库记账（元）','合计（元）','失负荷（kWh）'],table_rows)
    text(d,(68,1280),'4天、每策略384步；独立物理检查均通过。结果不代表全年排名。',28)
    foot(d,1550,'每一天均从同一初始库存独立运行；各次规划最大96步，尾部窗口逐渐缩短。');save(im,'02_新日期策略比较.png')

def budgets():
    rr=load('求解预算_固定状态.csv');budgets=[.25,2.,10.,60.];rows=[];values=[]
    for budget in budgets:
        r=[x for x in rr if f(x['budget_s'])==budget];g=[100*f(x['mip_gap']) for x in r if math.isfinite(f(x.get('mip_gap')))];med=statistics.median(g)
        values.append(g);rows.append([f'{budget:g} 秒',str(len(g)),f'{med:.2f}%',f'{max(g):.2f}%',str(sum(x['solver_status']=='optimal' for x in r))])
    im,d=canvas('求解时间：同一状态下是否接近最优？','诊断日每2小时取一个状态，共12个；预测窗口均为完整96步。',1420)
    ymax=max(5.,math.ceil(max(max(g) for g in values)/10)*10);Y=axis(d,(170,260,1715,735),ymax,'主规划相对间隙（%）')
    ytarget=Y(1.)
    for x in range(170,1715,25):d.line((x,ytarget,min(x+13,1715),ytarget),fill=RED,width=2)
    text(d,(1695,ytarget-12),'1%停止目标',24,RED,'rb')
    for i,(budget,gaps) in enumerate(zip(budgets,values)):
        x=335+i*390;med=statistics.median(gaps)
        for j,value in enumerate(gaps):
            dx=(j%5-2)*14;y=Y(value);d.ellipse((x+dx-7,y-7,x+dx+7,y+7),fill=BLUE)
        d.line((x-62,Y(med),x+62,Y(med)),fill=ORANGE,width=6);text(d,(x,763),f'{budget:g}秒',30,INK,'mt')
    text(d,(68,832),'蓝点：各状态间隙；橙线：中位数。间隙与经济最优性有关，不是功率误差。',26,MUTED)
    table(d,68,890,[300,300,350,350,364],['预算','有可行解/12','间隙中位数','最大间隙','达到停止准则'],rows,rowh=64)
    foot(d,1420,'这些是相同输入的单次主规划；不能把它的间隙当作整段滚动调度的最优性证明。');save(im,'03_求解时间与间隙.png')

def stress():
    rows=load('强化压力测试汇总.csv');scenarios=['normal','renewable_drop_60pct','renewable_zero','forced_failure'];labels=['正常','风光持续下降60%','风光降为零','强制求解失败']
    methods=['C_tcn_blend_q50','D_raw_reserve'];im,d=canvas('压力测试：备用能否减少缺供？','2019年11月11日；两策略均给2秒预算、相同预测和初始库存。',1550)
    ymax=max(100.,math.ceil(max(f(r['unserved_kwh']) for r in rows)/500)*500);Y=axis(d,(170,265,1715,715),ymax,'失负荷电量（kWh）')
    for i,sc in enumerate(scenarios):
        x=250+i*370
        for j,m in enumerate(methods):
            r=next(r for r in rows if r['scenario']==sc and r['strategy']==m);value=f(r['unserved_kwh']);xx=x+j*105
            if value>0:d.rectangle((xx,Y(value),xx+80,715),fill=BLUE if j==0 else GREEN)
            text(d,(xx+40,Y(value)-14),f'{value:,.1f}',26,INK,'mb')
        text(d,(x+92,745),labels[i],27,INK,'mt')
    text(d,(68,811),'蓝色：TCN q50；绿色：TCN原始区间备用。',27)
    bounds=load('能量守恒_缺供下界.csv');table_rows=[]
    for sc,label in zip(scenarios,labels):
        a=next(r for r in rows if r['scenario']==sc and r['strategy']==methods[0]);b=next(r for r in rows if r['scenario']==sc and r['strategy']==methods[1])
        bound=next(r for r in bounds if r['file'].replace('\\','/')==f'强化压力测试/{sc}/C_tcn_blend_q50/dispatch.csv')
        table_rows.append([label,f"{f(a['unserved_kwh']):,.2f}",f"{f(b['unserved_kwh']):,.2f}",f"{f(bound['unserved_energy_lower_bound_kwh']):,.2f}"])
    table(d,68,880,[510,390,390,374],['情景','q50失负荷（kWh）','备用失负荷（kWh）','能量下界（kWh）'],table_rows,rowh=76)
    text(d,(68,1317),'保留原始预测、降低实际风光，模拟持续预测偏差；未重新训练或调参数。',27)
    foot(d,1550,'强制失败同时关闭主规划与当步修正，用于检查安全回退，不用于评价预测方法。');save(im,'04_压力测试.png')

def trajectories():
    frames=[]
    for m in ['C_tcn_blend_q50','D_raw_reserve']:
        frames.append(load(Path('强化压力测试/renewable_zero')/m/'dispatch.csv'))
    im,d=canvas('风光降为零：库存与缺供轨迹','同一实际输入、同一初始库存；蓝色q50、绿色备用，轨迹基本重合。',1560)
    p=json.loads((OUT/'输入/config/devices.json').read_text(encoding='utf-8'))
    panels=[('unserved_kw','客户未满足功率（kW）',0.,max(200.,math.ceil(max(f(r['unserved_kw']) for frame in frames for r in frame)*1.1/200)*200),None),
            ('battery_energy_end_kwh','电池库存（kWh）',0.,2000.,(300.,1900.)),('hydrogen_end_kg','氢库存（kg）',0.,600.,(60.,540.))]
    for pi,(key,label,ymin,ymax,limits) in enumerate(panels):
        y0=260+pi*380;y1=y0+255;Y=axis(d,(170,y0,1715,y1),ymax,label,ticks=4)
        if limits:
            for limit in limits:
                yy=Y(limit)
                for x in range(170,1715,25):d.line((x,yy,min(x+12,1715),yy),fill=RED,width=2)
        for frame,col in zip(frames,[BLUE,GREEN]):
            points=[(170+1545*(i+1)/96,Y(f(r[key]))) for i,r in enumerate(frame)];d.line(points,fill=col,width=4)
        for hour in [0,6,12,18,24]:text(d,(170+1545*hour/24,y1+13),f'{hour:02d}:00',24,MUTED,'mt')
    text(d,(68,1397),'红色虚线：设备库存上下限。首个库存点为00:15，即第一步执行结束。',25,MUTED)
    foot(d,1560,'这是15分钟平均值的离线回放；不能据此证明电压、频率或真实设备跟踪性能。');save(im,'05_压力情景轨迹.png')

def braking():
    rows=load('高功率回退轨迹.csv');audit=json.loads((R/'高功率回退_独立审计.json').read_text(encoding='utf-8'))
    im,d=canvas('高功率运行时失败：设备怎样安全降功率？','初始电池550kWh、氢75kg；上一步放电和燃料电池均为500kW。',1160)
    Y=axis(d,(170,260,1715,610),500,'功率（kW）')
    for j,key,col in [(0,'battery_discharge_kw',BLUE),(1,'unserved_kw',RED)]:
        points=[(170+1545*(i+1)/6,Y(f(r[key]))) for i,r in enumerate(rows)];d.line(points,fill=col,width=5)
        for x,y in points:d.ellipse((x-6,y-6,x+6,y+6),fill=col)
    for step in range(1,7):text(d,(170+1545*step/6,640),f'第{step}步',27,MUTED,'mt')
    text(d,(68,717),'蓝色：电池和燃料电池各自功率（两条完全重合）；红色：失负荷功率。',27)
    table(d,68,790,[500,580,584],['检查项','最终状态','解释'],[
        ['电池库存',f"{audit['terminal_battery_kwh']:.2f} kWh",'高于300kWh下限'],
        ['氢库存',f"{audit['terminal_hydrogen_kg']:.2f} kg",'高于60kg下限']],rowh=62)
    foot(d,1160,'逐步降至零保持库存和爬坡合法；设备停机后仍可能缺供，安全回退不等于保供。');save(im,'06_高功率安全回退.png')

def energy():
    rows=load('能量守恒_缺供下界.csv')
    r=next(r for r in rows if r['file'].replace('\\','/')=='强化压力测试/renewable_zero/C_tcn_blend_q50/dispatch.csv')
    L=f(r['load_kwh']);available=f(r['max_grid_kwh'])+f(r['initial_usable_battery_bus_kwh'])+f(r['initial_usable_hydrogen_electric_kwh'])
    im,d=canvas('没有标准答案，也能证明最低缺供量','2019年11月11日，风光降为零；依据能量守恒推导，与优化器无关。',1320)
    Y=axis(d,(170,270,1715,650),30000,'全天电量（kWh）')
    bars=[('用户总需求',L,BLUE),('可用能源的理论上限',available,GREEN),('必然不足',L-available,RED)]
    for i,(label,value,col) in enumerate(bars):
        x=310+i*490;d.rectangle((x,Y(value),x+175,650),fill=col)
        text(d,(x+87,Y(value)-20),f'{value:,.2f}',32,INK,'mb');text(d,(x+87,685),label,28,INK,'mt')
    components=[['最多购电','500kW × 24小时',f"{f(r['max_grid_kwh']):,.2f}"],
        ['初始电池可供电','(1000−300)kWh × 放电效率',f"{f(r['initial_usable_battery_bus_kwh']):,.2f}"],
        ['初始氢可发电','(300−60)kg × 17.875kWh/kg',f"{f(r['initial_usable_hydrogen_electric_kwh']):,.2f}"]]
    table(d,68,772,[420,840,404],['能源来源','计算依据','电量（kWh）'],components,rowh=64)
    text(d,(68,1080),f"程序缺供：{f(r['observed_unserved_kwh']):,.2f}kWh，与理论下界一致（容差内）。",31,GREEN)
    foot(d,1320,'这证明本场景总缺供电量达到模型内最小值；不证明费用最优、唯一调度或真实设备性能。');save(im,'07_最低缺供的数学证据.png')

if __name__=='__main__':
    proof();new_days();budgets();stress();trajectories();braking();energy();print('Seven SimSun PNG figures generated.')
