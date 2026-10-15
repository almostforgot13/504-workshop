import numpy as np

def fit_cqr(q, truth, mask=None, alpha=0.2):
    """Horizon-wise split CQR; temporal dependence prevents an iid guarantee.

    Nonnegative expansion keeps the original median within every interval.
    Only a held-out calibration split may be supplied here.
    """
    q,truth = np.asarray(q),np.asarray(truth)
    if q.ndim != 3 or q.shape[-1] != 3 or q.shape[:2] != truth.shape or not 0 < alpha < 1:
        raise ValueError('Bad CQR shapes or alpha')
    if not np.isfinite(q).all() or not np.isfinite(truth).all() or (np.diff(q,axis=-1)<0).any():
        raise ValueError('CQR input must be finite and ordered')
    mask = np.ones_like(truth,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    scores = np.maximum(q[:,:,0]-truth,truth-q[:,:,2])
    corrections,counts = [],[]
    for h in range(truth.shape[1]):
        s = np.sort(scores[mask[:,h],h])
        n = len(s)
        rank = int(np.ceil((n+1)*(1-alpha)))
        if n < 20 or rank > n:
            raise ValueError(f'Insufficient calibration targets at horizon {h+1}')
        corrections.append(float(max(0,s[rank-1])))
        counts.append(n)
    return {'alpha':alpha,'correction_kw':corrections,'counts':counts,
        'coverage_interpretation':'empirical single-interval marginal coverage; no joint trajectory or temporal iid guarantee'}

def apply_cqr(q, calibration):
    correction = np.asarray(calibration['correction_kw'])
    if correction.shape != (q.shape[1],) or not np.isfinite(correction).all() or (correction<0).any():
        raise ValueError('Invalid calibration corrections')
    return q[:,:,0]-correction, q[:,:,2]+correction

def metrics(q, truth, mask=None, interval=None, alpha=0.2):
    q,truth = np.asarray(q),np.asarray(truth)
    mask = np.ones_like(truth,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    lo,hi = (q[:,:,0],q[:,:,2]) if interval is None else interval
    e = truth-q[:,:,1]
    width = hi-lo
    score = width + 2/alpha*np.maximum(lo-truth,0) + 2/alpha*np.maximum(truth-hi,0)
    def summarize(m):
        if not m.any():
            raise ValueError('No targets to score')
        result = {'pairs':int(m.sum()),'MAE_kw':float(np.abs(e)[m].mean()),
            'RMSE_kw':float(np.sqrt((e[m]**2).mean())),
            'coverage':float(((truth>=lo)&(truth<=hi))[m].mean()),
            'width_kw':float(width[m].mean()),'interval_score_kw':float(score[m].mean())}
        for j,tau in enumerate([0.1,0.5,0.9]):
            error = truth-q[:,:,j]
            result[f'pinball_{tau}'] = float(np.maximum(tau*error,(tau-1)*error)[m].mean())
        return result
    overall = summarize(mask)
    by_horizon=[]
    for h in range(truth.shape[1]):
        m=np.zeros_like(mask);m[:,h]=mask[:,h]
        by_horizon.append({'horizon_step':h+1,**summarize(m)})
    return overall,by_horizon
