"""Shared-data task controls and complete synchronized host-to-host timing."""
import argparse,random,platform
from public_common import *
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir();init();t0=time.time();m,state=load_engine();tx,ty,vx,vy,plan=load_data();write(out/'data_plan.json',plan)
fits={};start=time.perf_counter()
with torch.no_grad():
 for k in range(0,len(plan['fit']),8):m.forward(tensor(tx[plan['fit'][k:k+8]],'npu:0'),'fit',fits=fits)
 m.fit_finish(fits)
torch.npu.synchronize();fit_seconds=time.perf_counter()-start;np.savez_compressed(out/'learned_parameters.npz',**m.arrays())
configs=[(x,2,'fp32') for x in ['fp32','amp','none','block','smooth','learned']]+[('block',s,p) for s in [2,4,8,'global'] for p in ['fp32','fp16'] if not(s==2 and p=='fp32')]
rows=[];preds={};precision_checks=[]
with torch.no_grad():
 for mode,s,precision in configs:
  name=f'{mode}_s{s}_{precision}';alllog=[];beg=time.time()
  for k in range(0,len(plan['confirmation']),8):
   ids=plan['confirmation'][k:k+8];logits=m.forward(tensor(vx[ids],'npu:0'),mode,s,precision);alllog.append(arr(logits))
  logits=np.concatenate(alllog);preds[name]=logits;label=vy[plan['confirmation']];pr=logits.argmax(1);logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None]
  rows.append(dict(name=name,mode=mode,s=s,precision=precision,n=len(label),correct=int((pr==label).sum()),accuracy=float((pr==label).mean()),cross_entropy=float(-logp[np.arange(len(label)),label].mean()),seconds=time.time()-beg))
  write(out/'task_metrics.json',rows);np.savez_compressed(out/'task_logits.npz',indices=np.array(plan['confirmation']),labels=label,**preds);print('task',name,rows[-1]['accuracy'],'seconds',round(time.time()-beg),flush=True)
 # Target-precision accuracy on development inputs, each fixed layer and block size.
 for idx in plan['development'][:10]:
  x=tensor(tx[[idx]],'npu:0')
  for bi in INTERFACES:
   h,skip,c=m.forward(x,'none',capture=bi);y=m.fp(c,h);z=m.quant(c,h)
   for ss in [2,4,8,'global']:
    sz=y.shape[-1] if ss=='global' else ss;exact=F.avg_pool2d(y,sz)
    for precision in ['fp32','fp16']:
     tt=m.target(c,h,sz,precision);cb,_=m.correction(c,h,z,s=ss,precision=precision)
     precision_checks.append(dict(index=int(idx),block=bi,s=ss,precision=precision,target_mse=mse(tt,exact),local_none_mse=mse(z,y),local_block_mse=mse(cb,y),target_error_lift_energy=mse(tt,exact)))
 write(out/'target_precision.json',precision_checks)
 # Repeat identical tensor executions before timing; distinguishes arithmetic resolution from nondeterminism.
 repeat=[]
 for mode in ['fp32','none','block']:
  x=tensor(vx[plan['confirmation'][:8]],'npu:0');outputs=[arr(m.forward(x,mode)) for _ in range(3)]
  repeat.append(dict(mode=mode,max_pairwise=float(max(np.max(np.abs(outputs[i]-outputs[0])) for i in [1,2]))))
 write(out/'repeatability.json',repeat)
 # Count deployment storage separately from this analysis process, which retains all configurations.
 weights=dict(fp32_conv_bytes=sum(c['w'].numel()*4+c['b'].numel()*4 for c in m.c),quantized_conv_bytes=sum(c['qw'].numel()+c['sw'].numel()*4 for c in m.c),classifier_bytes=(m.fw.numel()+m.fb.numel())*4,learned_bytes=sum(b.numel()*b.element_size() for b in m.beta.values()))
 write(out/'storage_and_fit.json',dict(**weights,fit_seconds=fit_seconds,calibration_images=len(plan['fit'])))
print('task evaluation complete; run public_cost.py separately for valid timing',flush=True)
