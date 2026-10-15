from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/processed/A_15min_virtual_2017_2019.csv'
HORIZON = 96
HISTORY = 672
QUANTILES = np.array([0.1, 0.5, 0.9])
BASE_FIELDS = ['issue_time', 'target_time', 'horizon_step', 'horizon_hours', 'q10_kw', 'q50_kw', 'q90_kw']

def config():
    return json.loads((ROOT / 'config/experiment.json').read_text(encoding='utf-8'))

def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_data(path=DEFAULT_DATA):
    d = pd.read_csv(path)
    needed = {'time', 'load_kw', 'pv_kw', 'wind_kw'}
    if not needed.issubset(d):
        raise ValueError(f'Missing input fields: {needed-set(d)}')
    d['time'] = pd.to_datetime(d.time, utc=True, errors='raise')
    if len(d) < HISTORY or d.time.duplicated().any() or not (d.time.diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError('Need unique continuous UTC 15-minute input and at least 7 days of history')
    if (d.time.dt.minute % 15).any() or d.time.dt.second.any() or d.time.dt.microsecond.any():
        raise ValueError('Input timestamps must be UTC interval starts on quarter hours')
    a = d[['load_kw','pv_kw','wind_kw']].to_numpy(dtype=np.float32)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Actual load/PV/wind must be finite and nonnegative')
    y = a[:,0] - a[:,1] - a[:,2]
    if 'net_load_kw' in d and not np.allclose(y, d.net_load_kw, atol=0.002):
        raise ValueError('Declared net load differs from load minus PV minus wind')
    return d, a, y

def origins(d, start, end, stride):
    t = pd.DatetimeIndex(d.time)
    candidates = np.arange(HISTORY, len(d)-HORIZON+1, stride)
    # The whole target window must stay inside the selected split.
    return candidates[(t[candidates] >= pd.Timestamp(start)) & (t[candidates+HORIZON-1] < pd.Timestamp(end))]

def calendar(times):
    times = pd.DatetimeIndex(times)
    hour = times.hour.to_numpy() + times.minute.to_numpy()/60
    week = times.dayofweek.to_numpy()
    day = times.dayofyear.to_numpy()
    return np.column_stack([np.sin(hour*np.pi/12), np.cos(hour*np.pi/12),
        np.sin(week*2*np.pi/7), np.cos(week*2*np.pi/7),
        np.sin(day*2*np.pi/365.25), np.cos(day*2*np.pi/365.25)]).astype(np.float32)

def valid_targets(d, oo):
    target = np.asarray(oo)[:,None] + np.arange(HORIZON)
    if 'quality_flag' not in d:
        return np.ones(target.shape, dtype=bool)
    flagged = d.quality_flag.astype(str).str.contains('reconstructed').to_numpy()
    return ~flagged[target]

def forecasts_frame(times, oo, q, calibrated=None):
    oo = np.asarray(oo)
    q = np.asarray(q)
    if q.shape != (len(oo),HORIZON,3) or not np.isfinite(q).all() or (np.diff(q,axis=-1)<0).any():
        raise ValueError('Forecasts must be finite ordered [origins,96,3]')
    target = oo[:,None]+np.arange(HORIZON)
    t = pd.DatetimeIndex(times)
    flat = q.reshape(-1,3)
    frame = pd.DataFrame({'issue_time':np.repeat(t[oo].astype(str),HORIZON),
        'target_time':t[target.ravel()].astype(str), 'horizon_step':np.tile(np.arange(1,97),len(oo)),
        'horizon_hours':np.tile(np.arange(1,97)/4,len(oo)),
        'q10_kw':flat[:,0], 'q50_kw':flat[:,1], 'q90_kw':flat[:,2]})
    if calibrated is not None:
        lower,upper = calibrated
        frame['lower_cal_kw'] = np.asarray(lower).ravel()
        frame['upper_cal_kw'] = np.asarray(upper).ravel()
    return frame

def read_forecasts(path):
    f = pd.read_csv(path)
    if list(f.columns[:7]) != BASE_FIELDS or len(f)%HORIZON:
        raise ValueError('Expected base seven columns and complete 96-point windows')
    issue = pd.to_datetime(f.issue_time, utc=True)
    target = pd.to_datetime(f.target_time, utc=True)
    steps = f.horizon_step.to_numpy()
    if not np.array_equal(steps,np.tile(np.arange(1,97),len(f)//96)):
        raise ValueError('Missing, duplicate or unsorted horizon steps')
    issues = issue.to_numpy().reshape(-1,96)
    if not (issues == issues[:,0,None]).all() or pd.Index(issues[:,0]).duplicated().any():
        raise ValueError('Each issue must have exactly one complete window')
    if not (target == issue + pd.to_timedelta((steps-1)*15,unit='min')).all():
        raise ValueError('Target offset mismatch')
    if not np.allclose(f.horizon_hours,steps/4):
        raise ValueError('Horizon hours mismatch')
    numeric = ['q10_kw','q50_kw','q90_kw']
    if ('lower_cal_kw' in f) != ('upper_cal_kw' in f):
        raise ValueError('Both calibration fields must be supplied together')
    if 'lower_cal_kw' in f:
        numeric += ['lower_cal_kw','upper_cal_kw']
        if (f.lower_cal_kw > f.q50_kw).any() or (f.upper_cal_kw < f.q50_kw).any():
            raise ValueError('Calibrated interval must contain the median')
    if not np.isfinite(f[numeric].to_numpy()).all() or (np.diff(f[['q10_kw','q50_kw','q90_kw']],axis=1)<0).any():
        raise ValueError('Invalid forecast values/order')
    return f

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()
