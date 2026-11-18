"""Supplementary risk check: 10,000 paths, same selected models and sampling seed."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['MPLBACKEND']='Agg'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd

experiment_dir=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(experiment_dir))
from sde_gan import SDEGAN
from evaluate_increment_ablations import GROUPS,risk_scores,terminal
output=Path(__file__).resolve().parent
meta=json.loads((experiment_dir/'data/preprocessing.json').read_text())
mu,std=np.asarray(meta['training_mean']),np.asarray(meta['training_std'])
frame=pd.read_csv(experiment_dir/'data/validation_returns.csv',parse_dates=['Date']).set_index('Date')[meta['stock_order']]
dates=pd.read_csv(experiment_dir/'data/window_dates.csv',parse_dates=['start_date','end_date'])
observed=np.stack([frame.loc[d.start_date:d.end_date].to_numpy() for d in dates.loc[dates.split.eq('validation')].itertuples()])
realized=-terminal(observed)
rows=[]
for label,folder in GROUPS.items():
 for seed in (0,1,2):
  model=SDEGAN.load(experiment_dir/'results/sde_gan'/folder/f'seed_{seed}')
  x=model.sample(10000,100)
  with np.load(output/f'{folder}_seed_{seed}.npz') as small:
   np.testing.assert_array_equal(x[:1000],small['standardized_log_returns'])
  paths=np.expm1(x.astype(np.float64)*std+mu)
  for row in risk_scores(paths,realized):
   rows.append({**row,'experiment':label,'seed':seed,'n_paths':10000,'sample_seed':100})
  print('Completed',label,seed,flush=True)
risk=pd.DataFrame(rows)
risk.to_csv(output/'validation_risk_scores_10000_paths.csv',index=False)
summary=risk.groupby(['experiment','confidence']).agg(
 mean_quantile_loss_pct=('quantile_loss_pct','mean'),
 min_quantile_loss_pct=('quantile_loss_pct','min'),max_quantile_loss_pct=('quantile_loss_pct','max'),
 mean_breach_rate_pct=('breach_rate_pct','mean'))
summary.to_csv(output/'mean_risk_scores_10000_paths.csv')
print(summary.to_string(),flush=True)
