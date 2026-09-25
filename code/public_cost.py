"""Corrected warm-cache timing, separate preparation and nonadditive components."""
import argparse,random,platform,os
os.environ['TORCH_CPP_LOG_LEVEL']='ERROR'
from public_common import *
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir();init();torch.npu.synchronize();start=time.perf_counter();m,state=load_engine();torch.npu.synchronize();preparation=time.perf_counter()-start;tx,ty,vx,vy,plan=load_data()
f=np.load(os.environ['L1_LEARNED_PARAMETERS']);m.beta={n:torch.tensor(f[n],dtype=torch.float32,device='npu:0') for n in f.files}
configs=[(x,2,'fp32') for x in ['fp32','amp','none','block','smooth','learned']]+[('block',s,p) for s in [2,4,8,'global'] for p in ['fp32','fp16'] if not(s==2 and p=='fp32')];costs=[];rng=random.Random(92501)
with torch.no_grad():
 for batch in [1,8]:
  raw=vx[plan['confirmation'][:batch]]
  def call(cfg):return m.forward(tensor(raw,'npu:0'),*cfg).cpu()
  for cfg in configs:
   m.merged={};torch.npu.synchronize();beg=time.perf_counter();call(cfg);torch.npu.synchronize();cold=time.perf_counter()-beg
   mbytes=sum(w.numel()*w.element_size() for w in m.merged.values())
   costs.append(dict(name=f'{cfg[0]}_s{cfg[1]}_{cfg[2]}',batch=batch,cold_total_seconds=cold,prepared_kernel_bytes=mbytes,groups_ms=[]))
  # Warm every configuration AFTER the final cold-cache reset. No kernel construction in measured groups.
  for cfg in configs:
   for _ in range(10):call(cfg)
  torch.npu.synchronize()
  for group in range(7):
   order=list(range(len(configs)));rng.shuffle(order)
   for i in order:
    cfg=configs[i];torch.npu.reset_peak_memory_stats();torch.npu.synchronize();beg=time.perf_counter()
    for _ in range(20):call(cfg)
    torch.npu.synchronize();elapsed=(time.perf_counter()-beg)/20*1000;r=next(r for r in costs if r['batch']==batch and r['name']==f'{cfg[0]}_s{cfg[1]}_{cfg[2]}');r['groups_ms'].append(elapsed);r['peak_allocated_bytes']=int(torch.npu.max_memory_allocated());r['resident_allocated_bytes']=int(torch.npu.memory_allocated())
   write(out/'timing.json',costs);print('timing',batch,group+1,flush=True)
 # Individual synchronized components explain launch/transfer costs; they are not summed as an end-to-end estimate.
 parts=[]
 for bi in INTERFACES:
  x=tensor(tx[plan['development'][:8]],'npu:0');h,skip,c=m.forward(x,'none',capture=bi);z=m.quant(c,h);target=m.target(c,h,2);delta=target-F.avg_pool2d(z,2)
  funcs={'native_main':lambda:m.quant(c,h),'target_merged_fp32':lambda:m.target(c,h,2),'target_explicit_fp32':lambda:m.target(c,h,2,merged=False),'statistics':lambda:target-F.avg_pool2d(z,2),'lift_add_cast':lambda:(z+lift(delta,2)).half().float(),'full_correction':lambda:m.op(c,h,'block')}
  for name,fn in funcs.items():
   for _ in range(10):fn()
   torch.npu.synchronize();values=[]
   for group in range(5):
    beg=time.perf_counter()
    for _ in range(30):fn()
    torch.npu.synchronize();values.append((time.perf_counter()-beg)*1000/30)
   parts.append(dict(block=bi,name=name,batch=8,groups_ms=values,median_ms=float(np.median(values))))
write(out/'components.json',parts);write(out/'preparation.json',dict(load_fold_quantize_weight_seconds=preparation,torch=torch.__version__,python=platform.python_version(),scope='CPU uint8 input through normalization, transfer, all computations and casts to CPU float32 logits. Batch latency, not per-image. First configuration call in a shared process includes target-kernel preparation; not a fair cold-start ranking; warm groups after all configurations warm.',memory_note='Process holds all configurations; whole-process peak is not a minimal deployment claim.',validity='Uses all-configuration prewarming and suppresses repeated backend diagnostic warnings before torch import. All configurations prewarmed before measurements; task evaluation and fitting are separate.'))
print('cost complete',flush=True)
