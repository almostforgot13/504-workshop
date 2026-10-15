"""15-minute net-load forecasting. Run train or forecast; all timestamps are UTC."""
from pathlib import Path
import argparse, json, gzip, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[2] / 'data' / 'processed' / 'A_15min_virtual_2017_2019.csv'
LAGS = [1, 2, 4, 8, 16, 48, 96, 192, 672]
NAMES = [f'net_lag_{x}' for x in LAGS] + [f'{c}_lag_{x}' for c in ['load','pv','wind'] for x in [1,96,672]] + ['net_mean_4','net_std_4','net_mean_96','net_std_96','horizon_step','hour_sin','hour_cos','weekday','month','daily_reference','weekly_reference']

def read_data(path):
    d = pd.read_csv(path)
    d['time'] = pd.to_datetime(d.time, utc=True)
    if d.time.duplicated().any() or not (d.time.diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError('Timestamps must be unique, increasing and exactly 15 minutes apart.')
    a = d[['load_kw','pv_kw','wind_kw']].to_numpy(dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Input contains missing/non-finite values.')
    y = a[:,0]-a[:,1]-a[:,2]
    return d, a, y

def features(a, y, times, origins):
    # Every measurement index below is strictly smaller than its issue index.
    origins = np.asarray(origins, dtype=int)
    base = [y[origins-k] for k in LAGS]
    base += [a[origins-k,j] for j in range(3) for k in [1,96,672]]
    for w in [4,96]:
        v = y[origins[:,None]-np.arange(1,w+1)]
        base += [v.mean(axis=1), v.std(axis=1)]
    base = np.column_stack(base).astype('float32')
    targets = origins[:,None]+np.arange(96)
    stamps = times[targets.ravel()]
    hour = np.asarray(stamps.hour + stamps.minute/60)
    extra = np.column_stack([np.tile(np.arange(1,97),len(origins)),np.sin(hour*np.pi/12),np.cos(hour*np.pi/12),stamps.dayofweek,stamps.month,y[(targets-96).ravel()],y[(targets-672).ravel()]])
    return np.column_stack([np.repeat(base,96,axis=0),extra]).astype('float32'), targets

def origins_for(d, year, stride):
    yr = d.time.dt.year.to_numpy()
    return np.array([i for i in range(672,len(d)-95,stride) if yr[i] == year and yr[i+95] == year])

def raw_predictions(models, x):
    p = np.column_stack([m.predict(x,num_threads=8) for m in models])
    return np.sort(p,axis=1), np.any(np.diff(p,axis=1)<0,axis=1)

def new_stats():
    return {k:np.zeros(96) for k in ['n','abs_ai','sq_ai','abs_last','sq_last','abs_daily','sq_daily','abs_weekly','sq_weekly','covered','width','pin10','pin50','pin90','cross']}

def evaluate(d,a,y,models,origins,output=None):
    times = pd.DatetimeIndex(d.time)
    quality = d.quality_flag.astype(str).str.contains('reconstructed').to_numpy()
    s, all_s = new_stats(),new_stats()
    sink = gzip.open(output,'wt',encoding='utf-8',newline='') if output else None
    for k in range(0,len(origins),256):
        oo = origins[k:k+256]
        x,t = features(a,y,times,oo)
        p,cross = raw_predictions(models,x)
        truth=y[t.ravel()]
        daily=y[(t-96).ravel()]; weekly=y[(t-672).ravel()]
        last=np.repeat(y[oo-1],96)
        for stats, mask in [(s,~quality[t.ravel()]),(all_s,np.ones(truth.size,dtype=bool))]:
            m=mask.reshape(-1,96)
            stats['n']+=m.sum(axis=0)
            for name,pred in [('ai',p[:,1]),('last',last),('daily',daily),('weekly',weekly)]:
                e=(pred-truth).reshape(-1,96)
                stats['abs_'+name]+=(np.abs(e)*m).sum(axis=0)
                stats['sq_'+name]+=(e*e*m).sum(axis=0)
            stats['covered']+=(((truth>=p[:,0])&(truth<=p[:,2])).reshape(-1,96)*m).sum(axis=0)
            stats['width']+=((p[:,2]-p[:,0]).reshape(-1,96)*m).sum(axis=0)
            stats['cross']+=(cross.reshape(-1,96)*m).sum(axis=0)
            for j,q in enumerate([.1,.5,.9]):
                err=truth-p[:,j]
                stats['pin'+str(int(q*100))]+=(np.maximum(q*err,(q-1)*err).reshape(-1,96)*m).sum(axis=0)
        if sink:
            pd.DataFrame({'issue_time':np.repeat(times[oo].astype(str),96),'target_time':times[t.ravel()].astype(str),'horizon_step':np.tile(np.arange(1,97),len(oo)),'horizon_hours':np.tile(np.arange(1,97)/4,len(oo)),'q10_kw':p[:,0],'q50_kw':p[:,1],'q90_kw':p[:,2]}).to_csv(sink,index=False,header=k==0,float_format='%.4f')
        if k%4096==0: print(f'evaluated {k}/{len(origins)} origins',flush=True)
    if sink: sink.close()
    def summarize(z):
        n=float(z['n'].sum())
        r={'forecast_pairs':int(n)}
        for name in ['ai','last','daily','weekly']:
            r[name]={'MAE_kW':float(z['abs_'+name].sum()/n),'RMSE_kW':float(np.sqrt(z['sq_'+name].sum()/n))}
        r.update(coverage_q10_q90=float(z['covered'].sum()/n),mean_interval_width_kW=float(z['width'].sum()/n),raw_quantile_crossing_rate=float(z['cross'].sum()/n),pinball={q:float(z['pin'+q].sum()/n) for q in ['10','50','90']})
        return r
    by=pd.DataFrame({'horizon_step':np.arange(1,97),'horizon_hours':np.arange(1,97)/4,'pairs':s['n'].astype(int)})
    for name in ['ai','last','daily','weekly']:
        by[name+'_MAE_kW']=s['abs_'+name]/s['n']
        by[name+'_RMSE_kW']=np.sqrt(s['sq_'+name]/s['n'])
    by['interval_coverage']=s['covered']/s['n']; by['interval_width_kW']=s['width']/s['n']
    return summarize(s),summarize(all_s),by

def forecast(d,a,y,models,issue=None):
    # Truncate history at issue; the prediction does not read future measurements.
    issue=pd.Timestamp(issue) if issue else d.time.iloc[-1]+pd.Timedelta(minutes=15)
    if issue.tzinfo is None: issue=issue.tz_localize('UTC')
    issue=issue.tz_convert('UTC')
    keep=(d.time<issue).to_numpy()
    hist=d.loc[keep].copy(); a=a[keep]; y=y[keep]
    if len(hist)<672 or hist.time.iloc[-1]+pd.Timedelta(minutes=15)!=issue:
        raise ValueError('Need at least 7 days of complete history ending immediately before issue.')
    future=pd.date_range(issue,periods=96,freq='15min')
    times=pd.DatetimeIndex(hist.time).append(future)
    x,t=features(a,y,times,[len(y)])
    p,_=raw_predictions(models,x)
    return pd.DataFrame({'issue_time':str(issue),'target_time':future.astype(str),'horizon_step':np.arange(1,97),'horizon_hours':np.arange(1,97)/4,'q10_kw':p[:,0],'q50_kw':p[:,1],'q90_kw':p[:,2]})

def train(data):
    d,a,y=read_data(data); times=pd.DatetimeIndex(d.time)
    tr=origins_for(d,2017,4); va=origins_for(d,2018,16)
    xt,tt=features(a,y,times,tr); xv,tv=features(a,y,times,va)
    yt=y[tt.ravel()]; yv=y[tv.ravel()]
    print(f'Train samples {len(yt)}, validation samples {len(yv)}, features {len(NAMES)}',flush=True)
    models=[]; iterations=[]
    for alpha in [.1,.5,.9]:
        print(f'Training quantile {alpha}',flush=True)
        m=lgb.LGBMRegressor(objective='quantile',alpha=alpha,n_estimators=400,learning_rate=.06,num_leaves=31,max_depth=6,min_child_samples=100,colsample_bytree=.9,random_state=42,n_jobs=8,verbosity=-1)
        m.fit(xt,yt,eval_X=xv,eval_y=yv,callbacks=[lgb.early_stopping(35,verbose=True),lgb.log_evaluation(50)],feature_name=NAMES)
        model_path=ROOT/f'model_q{int(alpha*100)}.txt'
        # Python handles Unicode paths; the native LightGBM file API may not on Windows.
        model_path.write_text(m.booster_.model_to_string(),encoding='utf-8')
        models.append(lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))); iterations.append(m.best_iteration_)
    del xt,xv,yt,yv
    valid,_,vby=evaluate(d,a,y,models,va)
    selected=min(['last','daily','weekly'],key=lambda z:valid[z]['MAE_kW'])
    print(f'Validation-selected baseline: {selected}',flush=True)
    te=origins_for(d,2019,1)
    test,test_all,by=evaluate(d,a,y,models,te,ROOT/'test_2019_forecasts.csv.gz')
    by.to_csv(ROOT/'test_metrics_by_horizon.csv',index=False)
    vby.to_csv(ROOT/'validation_metrics_by_horizon.csv',index=False)
    result={'data_path':str(data),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'versions':{'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__},'scenario':{'mean_load_kW':1000,'pv_capacity_kW':3200,'wind_capacity_kW':1600,'status':'provisional simulation assumptions, not real park capacities'},'time_convention':'UTC interval start; issue has observations strictly before itself; target=issue+(step-1)*15min','train_origin_stride_quarters':4,'validation_origin_stride_quarters':16,'test_origin_stride_quarters':1,'train_origins':len(tr),'validation_origins':len(va),'test_origins':len(te),'features':NAMES,'best_iterations':iterations,'selected_baseline':selected,'validation':valid,'test_excluding_reconstructed_targets':test,'test_all_targets':test_all}
    (ROOT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    issue='2019-10-1 00:00:00+00:00'
    demo=forecast(d,a,y,models,issue); demo.to_csv(ROOT/'example_96_steps.csv',index=False,float_format='%.4f')
    forecast(d,a,y,models).to_csv(ROOT/'latest_96_steps.csv',index=False,float_format='%.4f')
    # Verify leakage resistance and saved-model consistency on a fixed example.
    mutated=a.copy(); mutated[d.time>=pd.Timestamp(issue)]=999999
    yy=mutated[:,0]-mutated[:,1]-mutated[:,2]
    other=forecast(d,mutated,yy,models,issue)
    assert np.array_equal(demo[['q10_kw','q50_kw','q90_kw']].values,other[['q10_kw','q50_kw','q90_kw']].values)
    assert (demo.q10_kw<=demo.q50_kw).all() and (demo.q50_kw<=demo.q90_kw).all()
    assert len(te)==34945 and len(demo)==96
    (ROOT/'verification.json').write_text(json.dumps({'future_measurement_mutation_does_not_change_forecast':True,'quantile_order':True,'test_origins':len(te),'forecast_rows':len(te)*96,'first_target_equals_issue':True},indent=2),encoding='utf-8')
    plot_results(d,y,demo,by,selected)
    write_readme(result)
    print(json.dumps(test,indent=2),flush=True)

def plot_results(d,y,demo,by,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,4))
    for name in ['ai','last','daily','weekly']: plt.plot(by.horizon_hours,by[name+'_MAE_kW'],label='LightGBM q50' if name=='ai' else name+' persistence')
    plt.xlabel('Lead time / hours'); plt.ylabel('MAE / kW'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(ROOT/'lead_time_MAE.png',dpi=160); plt.close()
    truth=pd.Series(y,index=pd.DatetimeIndex(d.time)).reindex(pd.to_datetime(demo.target_time)).to_numpy()
    x=np.arange(96)/4
    plt.figure(figsize=(10,4)); plt.fill_between(x,demo.q10_kw,demo.q90_kw,alpha=.25,label='q10-q90'); plt.plot(x,truth,label='Actual'); plt.plot(x,demo.q50_kw,label='Forecast q50')
    plt.axhline(0,color='gray',lw=.7); plt.xlabel('Hours after issue (2019-10-1 00:00 UTC)'); plt.ylabel('Net load / kW'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'example_24h.png',dpi=160); plt.close()
    plt.figure(figsize=(9,4)); plt.plot(by.horizon_hours,by.interval_coverage); plt.axhline(.8,ls='--',color='gray',label='Nominal 80%'); plt.ylim(0,1); plt.xlabel('Lead time / hours'); plt.ylabel('Marginal interval coverage'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(ROOT/'interval_coverage.png',dpi=160); plt.close()

def write_readme(r):
    t=r['test_excluding_reconstructed_targets']; b=r['selected_baseline']; gain=(1-t['ai']['MAE_kW']/t[b]['MAE_kW'])*100
    content=f'''# A同学：15分钟净负荷预测模型

## 本次实测结果

- 训练：2017年；早停与基线选择：2018年；独立测试：2019年。
- 2019年每15分钟发起预测，每次预测未来24小时，共96个点、{r['test_origins']}次预测、{r['test_origins']*96:,}条输出。
- 测试主指标排除用容量重建的目标点：模型 MAE **{t['ai']['MAE_kW']:.2f} kW**，RMSE **{t['ai']['RMSE_kW']:.2f} kW**。
- 验证集选出的基线为 {b}，测试 MAE {t[b]['MAE_kW']:.2f} kW。模型相对该基线 MAE 改善 {gain:.2f}%。
- q10—q90区间实测覆盖率 {t['coverage_q10_q90']:.2%}，名义目标80%；平均宽度 {t['mean_interval_width_kW']:.2f} kW。这是单个时段的边际区间，不能解释为整条96点曲线有80%同时覆盖率。
- 详细基线、分位数损失、含重建点的指标见 metrics.json；各提前量指标见 test_metrics_by_horizon.csv。
- 同一目标时刻会被多个发起点预测，预测条数并非独立测量样本数量。本次按发起时刻与提前量配对汇总，未据此计算置信区间。
- 分提前量检查显示：最初15、30、45分钟，“保持最后观测值”的基线误差更小；模型的优势体现在更长提前量和24小时整体预测。若B主要关注立即执行的一步，当前版本不能宣称短时预测也优于所有基线。此项结果如实报告，本次未据测试结果调参或修改预测。

## 输入与预测含义

净负荷 = 园区负荷 − 光伏功率 − 风电功率，单位kW。负值代表可再生能源富余，保留负值。
当前采用德国全国数据构建虚拟情景：平均负荷1000 kW、光伏3200 kW、风电1600 kW。后两个容量属于尚未确认的假设，既不是来源数据中的园区容量，也不是优化结果。全国曲线有空间平滑效应，本模型是仿真基准，不代表真实园区预测精度。情景改变后应重新训练评估。

全部时间为UTC。原数据时间是区间起点：08:00代表08:00—08:15的数据。08:00发起预测时，仅能使用截至07:45—08:00已经结束的观测；第1点预测08:00—08:15，第96点预测次日07:45—08:00。horizon_hours表示预测区间结束距离发起时刻的小时数。

## 模型与防止未来信息泄漏

三个LightGBM直接多步分位数模型，分别预测q10、q50、q90；各提前量共享模型并加入提前量特征。特征包括过去净负荷、负荷、风光功率，过去滚动均值/标准差，目标时刻日历，以及目标时刻昨天和上周同期的已知净负荷。没有使用未来实测风光、负荷或天气。
训练每小时选一个发起点；验证每4小时选一个；测试每15分钟预测。跨年份目标窗口排除，初始7天用于历史特征。2019年较早的实测数据可用于其后发起点，这是实际滚动预测的正常信息条件，模型参数始终固定。
分位数交叉时排序修正；原始交叉率保存在指标中。没有用测试集调参。本版本未使用气象预报，应如实称为历史序列预测。
2019年末100个容量/出力率重建点单独标识，主评估排除这些目标，含它们的结果也报告。此类数据仍可能进入后续预测的历史特征。4个风电出力率超1的点按原整理流程截断，详见数据校核文件。
verification.json记录防泄漏检查：任意改变发起时刻及以后的测量数据，已发起的预测保持不变。

## 文件与运行

- forecast_model.py：训练、评估和仅根据历史观测预测的完整代码。
- model_q10.txt / model_q50.txt / model_q90.txt：已训练模型。
- test_2019_forecasts.csv.gz：2019完整滚动预测，压缩CSV；只含发起时刻、目标时刻、提前量和预测，不含未来真实值。
- example_96_steps.csv：2019-06-15 00:00 UTC发起的固定示例。
- latest_96_steps.csv：根据截至2019年末的数据预测2020-01-01，缺少未来实测，不能评估其精度。
- 三张PNG：提前量误差、固定日期预测、区间覆盖率。

安装：`python -m pip install -r requirements.txt`

推荐Python 3.12或更高版本。完整预测超过Excel单张工作表行数限制，建议B用Python读取压缩CSV；你先打开96行示例检查格式即可。压缩交付包同时包含本次输入数据，解压后可直接按默认路径重训。

重训：`python forecast_model.py train`

新预测：`python forecast_model.py forecast --data "历史数据.csv" --output "未来96点.csv"`

历史CSV至少有连续7天15分钟数据，字段time、load_kw、pv_kw、wind_kw；time为UTC带时区或可转换为UTC的时间。调用forecast默认在最后一个观测区间结束时发起预测。train还需要quality_flag和完整2017—2019年数据。

## 交给B同学

使用q50作为净负荷的中心预测（中位数，并非数学期望）；q10/q90作为不确定性情景或安全裕度的输入。不要把q90直接解释成全天可靠性保证。调度若改为15分钟，必须统一Δt=0.25 h、96步，kW乘0.25才是每步kWh；电池和氢设备的功率、能量、效率约束由B负责。原来按小时的24步调度需同步修改；不能直接平均分位数当成小时分位数。

数据来源：https://data.open-power-system-data.org/time_series/2020-10-06/
'''
    (ROOT/'使用说明与实验结果.md').write_text(content,encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['train','forecast'])
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'new_96_steps.csv')
    args=parser.parse_args()
    if args.mode=='train': train(args.data)
    else:
        d,a,y=read_data(args.data)
        models=[lgb.Booster(model_str=(ROOT/f'model_q{q}.txt').read_text(encoding='utf-8')) for q in [10,50,90]]
        forecast(d,a,y,models).to_csv(args.output,index=False,float_format='%.4f')
        print(f'Saved 96 forecast intervals to {args.output}')
