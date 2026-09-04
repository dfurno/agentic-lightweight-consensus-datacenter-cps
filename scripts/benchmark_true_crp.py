#!/usr/bin/env python3
"""Latency and separately instrumented memory benchmark of true CRP."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, time, tracemalloc
from pathlib import Path
# Apply the frozen request before NumPy (and its BLAS runtime) is loaded.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
 os.environ[_name]=os.environ.get('CRP_BENCHMARK_THREADS','1')
import numpy as np, pandas as pd, yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.consensus.crp import CRPState, crp_consensus
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(x,c,state=None): return crp_consensus(x,CRPState() if state is None else state,**c)
def state_for(profile,x,c,initial_ewma=None):
 state=CRPState()
 if profile=='stress':
  state.ewma=np.asarray(initial_ewma,dtype=float).copy()
  state.below_streak=np.zeros_like(x)
 return state
def frozen_stress(n,c,seed):
 """Deterministically materialize the first input reaching configured max rounds."""
 rng=np.random.default_rng(seed+n)
 for attempt in range(1,10001):
  x=rng.normal(30,rng.uniform(.5,12),n)
  ewma=x+rng.normal(0,rng.uniform(0,15),n)
  result=call(x,c,state_for('stress',x,c,ewma))
  if result.rounds==c['max_rounds']:
   return x,ewma,attempt
 raise RuntimeError(f'no maximum-round stress input found for n={n}')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise SystemExit('output exists')
 cfg=yaml.safe_load(a.config.read_text())
 requested=str(cfg['threads'])
 if any(os.environ.get(k)!=requested for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')): raise SystemExit('set CRP_BENCHMARK_THREADS before process start')
 rng=np.random.default_rng(cfg['seed']); rows=[]; inputs=[]
 for n in cfg['sizes']:
  convergent=rng.normal(30,.5,n)
  # Frozen separated reporters plus a centre cluster exercise feedback rounds.
  stress,stress_ewma,attempt=frozen_stress(n,cfg['crp'],cfg['stress_seed'])
  inputs.append({'n':n,'profiles':{'convergent':{'values':convergent.tolist()},'stress':{'values':stress.tolist(),'initial_ewma':stress_ewma.tolist(),'deterministic_constructor_attempt':attempt}},'sha256':hashlib.sha256(convergent.tobytes()+stress.tobytes()+stress_ewma.tobytes()).hexdigest()})
  for profile,x,initial_ewma in [('convergent',convergent,None),('stress',stress,stress_ewma)]:
   for _ in range(cfg['warmup']): call(x,cfg['crp'],state_for(profile,x,cfg['crp'],initial_ewma))
   samples=[]; rounds=[]; conv=[]; exclusions=[]
   for _ in range(cfg['latency_repetitions']):
    start=time.perf_counter_ns(); result=call(x,cfg['crp'],state_for(profile,x,cfg['crp'],initial_ewma)); samples.append(time.perf_counter_ns()-start); rounds.append(result.rounds); conv.append(result.converged); exclusions.append(int(result.excluded.sum()))
   peaks=[]
   for _ in range(cfg['memory_repetitions']):
    tracemalloc.start(); call(x,cfg['crp'],state_for(profile,x,cfg['crp'],initial_ewma)); _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); peaks.append(peak)
   rows.append({'profile':profile,'n':n,'latency_repetitions':len(samples),'median_latency_us':statistics.median(samples)/1000,'p95_latency_us':float(np.percentile(samples,95))/1000,
   'min_latency_us':min(samples)/1000,'max_latency_us':max(samples)/1000,'median_rounds':statistics.median(rounds),'convergence_rate':float(np.mean(conv)),
   'median_exclusions':statistics.median(exclusions),'maximum_achieved_rounds':max(rounds),'configured_max_rounds':cfg['crp']['max_rounds'],
   'memory_repetitions':len(peaks),'median_peak_tracemalloc_bytes':statistics.median(peaks),'max_peak_tracemalloc_bytes':max(peaks),'latency_over_n_squared_ns':statistics.median(samples)/(n*n)})
 a.output.mkdir(parents=True); pd.DataFrame(rows).to_csv(a.output/'benchmark.csv',index=False); (a.output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
 cpu=Path('/proc/cpuinfo').read_text(); model=next((x.split(':',1)[1].strip() for x in cpu.splitlines() if x.startswith('model name')),None)
 commit=subprocess.run(['git','rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip(); prov={'run_id':cfg['run_id'],'source_code_commit':commit,'command':' '.join(sys.argv),'config_sha256':sha(a.config),'runner_sha256':sha(__file__),'inputs_sha256':sha(a.output/'inputs.json'),'python':platform.python_version(),'interpreter':sys.executable,'numpy':np.__version__,'pandas':pd.__version__,'pyyaml':yaml.__version__,'cpu_model':model,'vm_logical_cpu_count':os.cpu_count(),'host_declared_physical_cores':60,'host_declared_hardware_threads':120,'thread_limit_requested_before_numpy_import':True,'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},'latency_memory_separate':True,'llm_calls':0,'gpu_calls':0,'energy_measured':False,'tokens_measured':False}; (a.output/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
if __name__=='__main__': main()
