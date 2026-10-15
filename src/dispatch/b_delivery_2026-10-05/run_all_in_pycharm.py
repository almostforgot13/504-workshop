"""Open this folder in PyCharm and run this file for all four strategies."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'code'
ACTUAL=ROOT/'data/A_15min_virtual_2017_2019.csv'
FORECAST=ROOT/'data/A_48h_forecasts.csv'
START='2019-06-15T00:00:00Z'
STEPS=192


if __name__=='__main__':
    environment=os.environ.copy()
    environment['PYTHONIOENCODING']='utf-8'
    environment['AIC_ENERGY_SOURCE_CSV']=str(ACTUAL)
    result_dir=ROOT/'results'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(CODE),'-p','test_*.py','-v'],check=True,env=environment)
    subprocess.run([sys.executable,str(CODE/'dispatch.py'),'--actual',str(ACTUAL),
        '--forecast',str(FORECAST),'--start',START,'--steps',str(STEPS),
        '--strategy','all','--output-dir',str(result_dir)],check=True,env=environment)
    subprocess.run([sys.executable,str(CODE/'validate_dispatch.py'),'--actual',str(ACTUAL),
        '--results-dir',str(result_dir)],check=True,env=environment)
    print('Completed and independently verified:',result_dir)
