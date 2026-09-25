"""Frozen single-interface prediction followed by natural FP32 and INT8-led outcomes."""
import argparse
from public_common import *
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir();init();m,state=load_engine();tx,ty,vx,vy,plan=load_data();write(out/'data_plan.json',plan)
rows=[];cal=[];budget_start=time.time();prediction_file=out/'predictions_before_outcomes.jsonl'
def projection(a):
 C,H,W=a.shape[-3:];return np.repeat(np.repeat(a.reshape(C,H//2,2,W//2,2).mean((2,4)),2,1),2,2).reshape(-1)
def signed(n,b,o):
 n=arr(n).reshape(-1);b=arr(b).reshape(-1);o=arr(o).reshape(-1);D=float(np.mean((n-b)*(n+b-2*o)));T=float(np.mean((n-o)**2+(b-o)**2));eps=np.finfo(float).eps;tau=64*eps*float(np.mean((abs(n)+abs(b)+2*abs(o))**2))
 return dict(D=D,T=T,sign=int(D>tau)-int(D<-tau),rounding_scale=tau)
for role,ids,xx,labels in [('calibration',plan['calibration_mechanism'],tx,ty),('confirmation',plan['confirmation_mechanism'],vx,vy)]:
 for si,idx in enumerate(ids):
  x=tensor(xx[[idx]],'npu:0');label=int(labels[idx])
  with torch.no_grad():full=m.forward(x)
  for bi in INTERFACES:
   began=time.time()
   with torch.no_grad():
    h,skip,c=m.forward(x,'none',capture=bi);y=m.fp(c,h);zraw=m.quant(c,h);z=zraw.half().float();cb,d=m.correction(c,h,zraw);cb=cb.half().float();e=arr(z-y).reshape(-1);r=arr(cb-y).reshape(-1)
   yy=y.detach().clone().requires_grad_(True);oo=m.suffix(yy,skip,bi);J=[]
   for k in range(10):J.append(arr(torch.autograd.grad(oo[0,k],yy,retain_graph=k<9)[0]).reshape(-1))
   J=np.stack(J);Je=J@e;Jr=J@r;Ja=Je-Jr;u=np.linalg.norm(Ja);v=np.linalg.norm(Jr);energy=float(np.mean(Ja*Ja));cross=float(2*np.mean(Ja*Jr));L=float(np.mean(Je*Je-Jr*Jr));diag=np.mean(J*J,axis=0);diag_gain=float(np.sum(diag*(e*e-r*r)));trace_gain=float(diag.mean()*np.sum(e*e-r*r));local_gain=float(np.mean(e*e-r*r))
   ae=projection(e.reshape(y.shape));rp=e-ae;Jpa=J@ae;ca=J.T@Jpa;ca=ca-projection(ca.reshape(y.shape));rnorm=float(np.linalg.norm(rp));u2=float(Jpa@Jpa);cn=float(np.linalg.norm(ca));derivative_seconds=time.time()-began
   alphas=[.25,1.,4.] if role=='confirmation' and idx in plan['amplitude'] else [1.]
   for alpha in alphas:
    key=f'{role}_{idx}_b{bi}_a{alpha:g}';prediction=dict(case=key,role=role,index=int(idx),label=label,block=bi,layer=c['name'],alpha=alpha,linear_gain=L*alpha**2,energy_only=energy*alpha**2,cross=cross*alpha**2,diag_gain=diag_gain*alpha**2,trace_gain=trace_gain*alpha**2,local_gain=local_gain*alpha**2,rho=float(Ja@Jr/(u*v)) if u*v>0 else None,v_over_u=float(v/u) if u>0 else None,projected_residual_norm=rnorm,projected_Ja_norm2=u2,c_norm=cn,projection_cast_drift=float(np.linalg.norm(r-rp)),derivative_seconds=derivative_seconds,input_sha=hashlib.sha256(arr(h).tobytes()).hexdigest(),J_sha=hashlib.sha256(J.tobytes()).hexdigest())
    with prediction_file.open('a') as f:f.write(json.dumps(prediction)+'\n');f.flush();os.fsync(f.fileno())
    with torch.no_grad():
     yn=z if alpha==1 else y+alpha*(z-y);yb=cb if alpha==1 else y+alpha*(cb-y);o=m.suffix(y,skip,bi);n=m.suffix(yn,skip,bi);b=m.suffix(yb,skip,bi)
     D=signed(n,b,o);nerror=arr(n-o).reshape(-1);berror=arr(b-o).reshape(-1);qe=nerror-alpha*Je;qr=berror-alpha*Jr;B=float(2*np.linalg.norm(alpha*Je)*np.linalg.norm(qe)/10+np.mean(qe*qe)+2*np.linalg.norm(alpha*Jr)*np.linalg.norm(qr)/10+np.mean(qr*qr));assert abs(D['D']-prediction['linear_gain'])<=B+1e-12
     row=dict(**prediction,fp32_suffix=D,full_reference=signed(n,b,full),none_mse=mse(n,o),block_mse=mse(b,o),nonlinear_none=float(np.mean(qe*qe)),nonlinear_block=float(np.mean(qr*qr)),remainder_bound=B,none_correct=int(int(n.argmax())==label),block_correct=int(int(b.argmax())==label),label_ce_gain=float(F.cross_entropy(n,torch.tensor([label],device='npu:0'))-F.cross_entropy(b,torch.tensor([label],device='npu:0'))))
     if alpha==1:
      no=m.suffix(y,skip,bi,'none');nn=m.suffix(z,skip,bi,'none');nb=m.suffix(cb,skip,bi,'none');row['int8_suffix']=signed(nn,nb,no);row['int8_full_reference']=signed(nn,nb,full);row['int8_label_ce_gain']=float(F.cross_entropy(nn,torch.tensor([label],device='npu:0'))-F.cross_entropy(nb,torch.tensor([label],device='npu:0')))
      if bi<8:
       a0=F.relu(z+skip);a1=F.relu(cb+skip);q0,s0=m.codes(a0);q1,s1=m.codes(a1);qfixed,_=m.codes(a1,s0);nextc=m.blocks[bi+1][0];qn=m.quant(nextc,a0);qb=m.quant(nextc,a1)
       row['requant']=dict(dynamic_code_change=float((q0!=q1).float().mean()),fixed_scale_code_change=float((q0!=qfixed).float().mean()),scale_ratio=float((s1/s0).item()),retained_input_change_mse=mse(q1.float()*s1,q0.float()*s0),prequant_change_mse=mse(a1,a0),next_layer_change_mse=mse(qb,qn),correction_over_step_quantiles=np.quantile(np.abs(arr(a1-a0)/arr(s0)),[.25,.5,.75,.95,1]).tolist(),saturation_none=float((q0.abs()==127).float().mean()),saturation_block=float((q1.abs()==127).float().mean()))
      np.savez_compressed(out/(key+'.npz'),y=arr(y),z=arr(z),corrected=arr(cb),Je=Je,Jr=Jr,oracle=arr(o),none=arr(n),block=arr(b),full=arr(full),int8_oracle=arr(no),int8_none=arr(nn),int8_block=arr(nb))
     rows.append(row)
   if role=='calibration':cal.append(dict(block=bi,residual_norm=rnorm))
  write(out/'rows.json',rows)
  if si%10==0:print(role,si+1,len(ids),'rows',len(rows),'seconds',round(time.time()-budget_start),flush=True)
 # Freeze R before any confirmation outcomes.
 if role=='calibration':
  R={str(b):float(sorted(r['residual_norm'] for r in cal if r['block']==b)[int(np.ceil(.95*40))-1]) for b in INTERFACES};write(out/'residual_bounds_before_confirmation.json',R)
R=json.loads((out/'residual_bounds_before_confirmation.json').read_text());cert=[]
for r in rows:
 if r['role']!='confirmation' or r['alpha']!=1:continue
 rad=R[str(r['block'])];lower=(r['projected_Ja_norm2']-2*rad*r['c_norm'])/10
 cert.append(dict(case=r['case'],block=r['block'],residual_covered=r['projected_residual_norm']<=rad,affine_nonnegative_candidate=lower>=0,lower=lower,actual_gain=r['fp32_suffix']['D'],projection_cast_drift=r['projection_cast_drift']))
write(out/'residual_coverage.json',cert);write(out/'runtime.json',dict(seconds=time.time()-budget_start,count=len(rows),scope='exact output-Jacobian diagnostic; not a low-cost online selector'))
print('completed',len(rows),flush=True)
