"""One lightweight TCN, original-style LightGBM, honest chronological evaluation."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .data import ROOT, HISTORY, HORIZON, calendar, config, dump_json, read_data, origins, valid_targets, forecasts_frame, sha256
from .calibration import fit_cqr, apply_cqr, metrics

LAGS=[1,2,4,8,16,48,96,192,672]

def lgbm_features(a,y,times,oo):
    oo=np.asarray(oo,dtype=int)
    base=[y[oo-k] for k in LAGS]
    base += [a[oo-k,j] for j in range(3) for k in [1,96,672]]
    for w in [4,96]:
        v=y[oo[:,None]-np.arange(1,w+1)]
        base += [v.mean(axis=1),v.std(axis=1)]
    target=oo[:,None]+np.arange(96)
    t=pd.DatetimeIndex(times)[target.ravel()]
    hour=t.hour.to_numpy()+t.minute.to_numpy()/60
    extra=np.column_stack([np.tile(np.arange(1,97),len(oo)),np.sin(hour*np.pi/12),np.cos(hour*np.pi/12),t.dayofweek,t.month,y[(target-96).ravel()],y[(target-672).ravel()]])
    return np.column_stack([np.repeat(np.column_stack(base),96,axis=0),extra]).astype(np.float32),target

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.pad=2*dilation
        self.convs=nn.ModuleList([nn.Conv1d(channels,channels,3,dilation=dilation,padding=self.pad) for _ in range(2)])
        self.dropout=nn.Dropout(0.1)
    def forward(self,x):
        z=x
        for conv in self.convs:
            z=self.dropout(torch.relu(conv(z)[:,:,:-self.pad]))
        return x+z

class QuantileTCN(nn.Module):
    def __init__(self,channels=12):
        super().__init__()
        self.project=nn.Conv1d(4,channels,1)
        self.blocks=nn.Sequential(*[CausalBlock(channels,2**i) for i in range(8)])
        # Receptive field 1021 intervals, covering all 672 historical intervals.
        self.head=nn.Sequential(nn.Linear(channels*2+9,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,history,future):
        z=self.blocks(self.project(history))
        context=torch.cat([z[:,:,-1],z.mean(dim=2)],dim=1)
        raw=self.head(torch.cat([context[:,None,:].expand(-1,96,-1),future],dim=-1))
        median=future[:,:,6]+raw[:,:,0]  # residual from yesterday, learned only on training targets
        lower=median-nn.functional.softplus(raw[:,:,1])
        upper=median+nn.functional.softplus(raw[:,:,2])
        return torch.stack([lower,median,upper],dim=-1)

class WindowDataset(Dataset):
    def __init__(self,d,a,y,oo,normalizer):
        self.oo=np.asarray(oo)
        raw=np.column_stack([a,y])
        self.history=((raw-np.asarray(normalizer['mean']))/np.asarray(normalizer['std'])).astype(np.float32)
        self.y=y
        self.scale=normalizer['target_scale']
        self.cal=calendar(d.time)
    def __len__(self):return len(self.oo)
    def __getitem__(self,i):
        o=int(self.oo[i]); target=o+np.arange(96)
        future=np.column_stack([self.cal[target],self.y[target-96]/self.scale,
            self.y[target-672]/self.scale,np.arange(1,97)/96]).astype(np.float32)
        return self.history[o-672:o].T.copy(),future,(self.y[target]/self.scale).astype(np.float32)

def pinball(q,y):
    error=y[:,:,None]-q
    tau=q.new_tensor([0.1,0.5,0.9])
    return torch.maximum(tau*error,(tau-1)*error).mean()

def predict_tcn(model,dataset,batch_size=128):
    predictions=[]
    model.eval()
    with torch.inference_mode():
        for h,f,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            predictions.append(model(h,f).numpy()*dataset.scale)
    return np.concatenate(predictions)

def predict_lgbm(models,a,y,times,oo,threads=4):
    parts=[]
    for start in range(0,len(oo),256):
        x,_=lgbm_features(a,y,times,oo[start:start+256])
        q=np.column_stack([m.predict(x,num_threads=threads) for m in models])
        parts.append(np.sort(q,axis=-1).reshape(-1,96,3))
    return np.concatenate(parts)

def choose_blend(q,y,oo,truth,mask):
    weights=np.ones(96)
    for h in range(3):
        errors=[]
        for weight in np.linspace(0,1,11):
            point=weight*q[:,h,1]+(1-weight)*y[oo-1]
            errors.append(np.abs(point-truth[:,h])[mask[:,h]].mean())
        weights[h]=float(np.linspace(0,1,11)[np.argmin(errors)])
    return weights

def blend(q,y,oo,weights):
    w=np.asarray(weights)[None,:,None]
    return w*q+(1-w)*y[np.asarray(oo)-1,None,None]

def train_all(data,epochs=None):
    import lightgbm as lgb
    cfg=config();d,a,y=read_data(data)
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    tr=origins(d,cfg['train_start'],cfg['train_end'],cfg['train_stride'])
    va=origins(d,cfg['selection_start'],cfg['selection_end'],cfg['validation_stride'])
    ca=origins(d,cfg['calibration_start'],cfg['calibration_end'],cfg['calibration_stride'])
    if min(len(tr),len(va),len(ca))<20:raise ValueError('Incomplete train/selection/calibration history')
    times=pd.DatetimeIndex(d.time)
    model_dir=ROOT/'models';model_dir.mkdir(exist_ok=True)
    xt,tt=lgbm_features(a,y,times,tr);xv,tv=lgbm_features(a,y,times,va)
    models=[]
    start=time.perf_counter()
    for tau in [0.1,0.5,0.9]:
        print(f'LightGBM quantile={tau}: {len(tr)} training origins',flush=True)
        model=lgb.train({'objective':'quantile','alpha':tau,'learning_rate':.06,'num_leaves':31,'max_depth':6,
            'min_data_in_leaf':100,'feature_fraction':.9,'seed':cfg['seed'],'num_threads':cfg['threads'],'verbosity':-1},
            lgb.Dataset(xt,label=y[tt.ravel()]),num_boost_round=400,
            valid_sets=[lgb.Dataset(xv,label=y[tv.ravel()])],callbacks=[lgb.early_stopping(35,verbose=False)])
        (model_dir/f'lgbm_q{int(tau*100)}.txt').write_text(model.model_to_string(),encoding='utf-8')
        models.append(model)
    lgb_seconds=time.perf_counter()-start
    del xt,xv
    train_rows=(d.time>=pd.Timestamp(cfg['train_start']))&(d.time<pd.Timestamp(cfg['train_end']))
    raw=np.column_stack([a,y])[train_rows]
    norm={'mean':raw.mean(axis=0).astype(float).tolist(),'std':np.maximum(raw.std(axis=0),1).astype(float).tolist(),'target_scale':float(max(y[train_rows].std(),1))}
    train_set=WindowDataset(d,a,y,tr,norm);val_set=WindowDataset(d,a,y,va,norm)
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(train_set,batch_size=cfg['batch_size'],shuffle=True,generator=generator)
    model=QuantileTCN(cfg['tcn_channels']);optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    best=float('inf');patience=0;history=[];best_epoch=0
    start=time.perf_counter()
    for epoch in range(epochs or cfg['tcn_epochs']):
        model.train();loss_sum=0
        for h,f,target in loader:
            optimizer.zero_grad();loss=pinball(model(h,f),target);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_sum+=float(loss.detach())*len(h)
        model.eval();val_loss=0
        with torch.inference_mode():
            for h,f,target in DataLoader(val_set,batch_size=cfg['batch_size']):
                val_loss+=float(pinball(model(h,f),target))*len(h)
        val_loss/=len(val_set)
        history.append({'epoch':epoch+1,'train_pinball_kw':loss_sum/len(train_set)*norm['target_scale'],'selection_pinball_kw':val_loss*norm['target_scale']})
        print(f'TCN epoch={epoch+1} train={history[-1]["train_pinball_kw"]:.3f} selection={history[-1]["selection_pinball_kw"]:.3f}',flush=True)
        if val_loss<best:
            best=val_loss;patience=0;best_epoch=epoch+1
            torch.save({'state_dict':model.state_dict(),'normalizer':norm,'channels':cfg['tcn_channels']},model_dir/'tcn.pt')
        else:
            patience+=1
            if patience>=cfg['tcn_patience']:break
    model,_=load_tcn()
    tcn_seconds=time.perf_counter()-start
    qv=predict_tcn(model,val_set);maskv=valid_targets(d,va);truthv=y[va[:,None]+np.arange(96)]
    weights=choose_blend(qv,y,va,truthv,maskv)
    qvb=blend(qv,y,va,weights)
    qlv=predict_lgbm(models,a,y,times,va,cfg['threads'])
    qc=blend(predict_tcn(model,WindowDataset(d,a,y,ca,norm)),y,ca,weights)
    truthc=y[ca[:,None]+np.arange(96)]
    cqr=fit_cqr(qc,truthc,valid_targets(d,ca),cfg['alpha'])
    dump_json(model_dir/'calibration.json',cqr)
    dump_json(model_dir/'fusion.json',{'weights':weights.tolist(),'selected_on':'2018 H1 only','first_three_horizons_only':True})
    summary={'data_sha256':sha256(data),'experiment_config':cfg,'train_origins':len(tr),'selection_origins':len(va),'calibration_origins':len(ca),
        'best_tcn_epoch':best_epoch,'tcn_training_seconds':tcn_seconds,'lgbm_training_seconds':lgb_seconds,'history':history,
        'selection_lgbm':metrics(qlv,truthv,maskv)[0],'selection_tcn_raw':metrics(qv,truthv,maskv)[0],
        'selection_tcn_blend':metrics(qvb,truthv,maskv)[0],
        'calibration_fit_diagnostic':metrics(qc,truthc,valid_targets(d,ca),apply_cqr(qc,cqr))[0],
        'versions':{'torch':torch.__version__,'lightgbm':lgb.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'test_tuning':False,'historical_2019_results_previously_seen':True,
        'calibration_diagnostic_is_out_of_sample_test':False}
    dump_json(model_dir/'training.json',summary)
    freeze={'data_sha256':sha256(data),'config_sha256':sha256(ROOT/'config/experiment.json'),
        'artifacts':{p.name:sha256(p) for p in sorted(model_dir.iterdir()) if p.is_file() and p.name!='freeze.json'},
        'source_artifacts':{p.name:sha256(p) for p in (ROOT/'energy_ai').glob('*.py') if p.name in ['data.py','forecast.py','calibration.py','dispatch.py','device_model.py','experiments.py']},
        'test_weeks':cfg['test_weeks'],'no_post_test_parameter_changes':True}
    dump_json(model_dir/'freeze.json',freeze)
    print('Frozen models, fusion and calibration ready.',flush=True)

def load_tcn():
    checkpoint=torch.load(ROOT/'models/tcn.pt',map_location='cpu',weights_only=True)
    model=QuantileTCN(checkpoint['channels']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint['normalizer']

def assert_frozen(data):
    frozen=json.loads((ROOT/'models/freeze.json').read_text(encoding='utf-8'))
    if sha256(data)!=frozen['data_sha256'] or sha256(ROOT/'config/experiment.json')!=frozen['config_sha256']:
        raise ValueError('Data/config changed after freeze; retrain and declare a new experiment')
    for name,digest in frozen['artifacts'].items():
        if sha256(ROOT/'models'/name)!=digest:raise ValueError(f'Frozen artifact changed: {name}')
    for name,digest in frozen.get('source_artifacts',{}).items():
        if sha256(ROOT/'energy_ai'/name)!=digest:raise ValueError(f'Frozen experiment implementation changed: {name}')
    return frozen

def export_weeks(data,starts=None,steps=672):
    import lightgbm as lgb
    cfg=config();assert_frozen(data);torch.set_num_threads(cfg['threads'])
    d,a,y=read_data(data);model,norm=load_tcn()
    weights=json.loads((ROOT/'models/fusion.json').read_text())['weights']
    cqr=json.loads((ROOT/'models/calibration.json').read_text())
    models=[lgb.Booster(model_str=(ROOT/f'models/lgbm_q{q}.txt').read_text()) for q in [10,50,90]]
    report={}
    for start in starts or cfg['test_weeks']:
        o=int(d.index[d.time==pd.Timestamp(start)][0]);oo=np.arange(o,o+steps)
        if oo[-1]+95>=len(d):raise ValueError('Not enough future targets for evaluation')
        ql=predict_lgbm(models,a,y,d.time,oo,cfg['threads'])
        qt=predict_tcn(model,WindowDataset(d,a,y,oo,norm));qb=blend(qt,y,oo,weights)
        interval=apply_cqr(qb,cqr);truth=y[oo[:,None]+np.arange(96)];mask=valid_targets(d,oo)
        key=pd.Timestamp(start).strftime('%Y%m%d')
        folder=ROOT/'results/forecasts'/key;folder.mkdir(parents=True,exist_ok=True)
        forecasts_frame(d.time,oo,ql).to_csv(folder/'lightgbm.csv.gz',index=False,float_format='%.6f')
        forecasts_frame(d.time,oo,qb,interval).to_csv(folder/'tcn_blend.csv.gz',index=False,float_format='%.6f')
        scores={}
        for name,q,inter in [('lgbm',ql,None),('tcn_raw',qt,None),('tcn_blend',qb,None),('tcn_cqr',qb,interval)]:
            score,by=metrics(q,truth,mask,inter,cfg['alpha']);scores[name]=score
            pd.DataFrame(by).to_csv(folder/f'{name}_by_horizon.csv',index=False)
        for name,pred in [('last',np.repeat(y[oo-1,None],96,axis=1)),('daily',y[oo[:,None]+np.arange(96)-96])]:
            scores[name]={'MAE_kw':float(np.abs(pred-truth)[mask].mean())}
        dump_json(folder/'metrics.json',scores);report[key]=scores
        print(f'Exported {key}: {steps} complete forecast windows',flush=True)
    dump_json(ROOT/'results/forecast_metrics.json',report)

def forecast_one(data,issue,model_name='tcn'):
    """Forecast only from completed history; works with a history-only input CSV."""
    import lightgbm as lgb
    d,a,y=read_data(data);issue=pd.Timestamp(issue)
    if issue.tzinfo is None:raise ValueError('Issue must explicitly contain a timezone')
    issue=issue.tz_convert('UTC');keep=(d.time<issue).to_numpy();d=d.loc[keep].reset_index(drop=True);a=a[keep];y=y[keep]
    if len(y)<672 or d.time.iloc[-1]+pd.Timedelta(minutes=15)!=issue:
        raise ValueError('Need seven continuous days of completed history ending at issue')
    future=pd.date_range(issue,periods=96,freq='15min');extended=d.copy()
    extended=pd.concat([extended,pd.DataFrame({'time':future})],ignore_index=True)
    # Future zero slots are never read as measurements; only calendars are accessed.
    aa=np.vstack([a,np.zeros((96,3),dtype=np.float32)]);yy=np.r_[y,np.zeros(96,dtype=np.float32)];oo=np.array([len(y)])
    if model_name=='tcn':
        model,norm=load_tcn();q=predict_tcn(model,WindowDataset(extended,aa,yy,oo,norm))
        w=json.loads((ROOT/'models/fusion.json').read_text())['weights'];q=blend(q,yy,oo,w)
        cqr=json.loads((ROOT/'models/calibration.json').read_text());inter=apply_cqr(q,cqr)
    else:
        models=[lgb.Booster(model_str=(ROOT/f'models/lgbm_q{q}.txt').read_text()) for q in [10,50,90]]
        q=predict_lgbm(models,aa,yy,extended.time,oo);inter=None
    return forecasts_frame(extended.time,oo,q,inter)
