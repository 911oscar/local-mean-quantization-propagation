"""CIFAR ResNet inference and native NPU quantization; explicit precision paths."""
import os
from pathlib import Path
root=Path(os.environ.get('L1_OUTPUT_DIR','runs'));cache=root/'cache/ascend';cache.mkdir(parents=True,exist_ok=True)
for key in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH']:os.environ[key]=str(cache)
for key in ['TE_PARALLEL_COMPILER','MAX_COMPILE_CORE_NUMBER','CANN_KNOWLEDGE_BANK_PROCESS_NUM']:os.environ[key]='1'
os.environ['TUNE_BANK_PATH']=str(cache/'aoe');(cache/'aoe').mkdir(exist_ok=True)
import torch,numpy as np,torch.nn.functional as F
import json,hashlib,time,tarfile,pickle,sys
ASSETS=Path(os.environ.get('L1_ASSET_DIR','assets'))
INTERFACES=[0,2,3,5,6,8]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def init(device='npu:0'):
 torch.set_num_threads(2);torch.manual_seed(92501)
 if device.startswith('npu'):
  import torch_npu
  torch.npu.set_device(device);torch.npu.config.allow_internal_format=False;torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
 return device

def load_data(assets=ASSETS):
 rows=json.loads((assets/'assets.json').read_text())
 for r in rows:assert sha(assets/r['name'])==r['sha256']
 with tarfile.open(assets/'cifar10.tar.gz') as t:
  def get(n):
   with t.extractfile('cifar-10-batches-py/'+n) as f:return pickle.load(f,encoding='bytes')
  tr=[get('data_batch_'+str(i)) for i in range(1,6)];te=get('test_batch')
 tx=np.concatenate([b[b'data'] for b in tr]).reshape(-1,3,32,32);ty=np.concatenate([b[b'labels'] for b in tr]);vx=te[b'data'].reshape(-1,3,32,32);vy=np.array(te[b'labels'])
 def choose(labels,prefix,n):
  return [i for c in range(10) for i in sorted(np.flatnonzero(labels==c).tolist(),key=lambda i:hashlib.sha256((prefix+str(i)).encode()).hexdigest())[:n]]
 ci=choose(ty,'L1R-cal-v1:',32);ti=choose(vy,'L1R-test-v1:',100)
 fit=[i for c in range(10) for i in ci[c*32:c*32+24]];dev=[i for c in range(10) for i in ci[c*32+24:c*32+32]]
 mc=[i for c in range(10) for i in ci[c*32:c*32+4]];mt=[i for c in range(10) for i in ti[c*100:c*100+12]];amp=[i for c in range(10) for i in ti[c*100:c*100+4]]
 return tx,ty,vx,vy,dict(fit=fit,development=dev,calibration_mechanism=mc,confirmation=ti,confirmation_mechanism=mt,amplitude=amp)
def tensor(a,device):
 x=torch.from_numpy(np.asarray(a,dtype=np.float32)/255).to(device)
 return (x-x.new_tensor([.4914,.4822,.4465])[None,:,None,None])/x.new_tensor([.2023,.1994,.2010])[None,:,None,None]
def lift(t,s):return t.repeat_interleave(s,-2).repeat_interleave(s,-1)
def mse(a,b=None):
 x=a.detach().cpu().double();x=x if b is None else x-b.detach().cpu().double();return float(x.square().mean())
def arr(x):return x.detach().cpu().double().numpy()
def expanded(w,s,q=1):
 e=w.new_zeros((*w.shape[:2],w.shape[-2]+q*(s-1),w.shape[-1]+q*(s-1)))
 for i in range(s):
  for j in range(s):e[:,:,i*q:i*q+w.shape[-2],j*q:j*q+w.shape[-1]]+=w/(s*s)
 return e
class Engine:
 def __init__(self,state,device='npu:0'):
  self.device=device;self.c=[];self.merged={};self.beta={}
  def folded(cp,bp,stride=1):
   w=state[cp+'.weight'].float();scale=state[bp+'.weight']/torch.sqrt(state[bp+'.running_var']+1e-5)
   b=state[bp+'.bias']-state[bp+'.running_mean']*scale;w=w*scale[:,None,None,None]
   obj={'w':w.to(device),'b':b.to(device),'stride':stride,'padding':w.shape[-1]//2,'name':cp}
   sw=w.abs().flatten(1).amax(1).clamp_min(1e-12)/127;obj['sw']=sw.to(device);obj['qw']=(w/sw[:,None,None,None]).round().clamp(-127,127).to(torch.int8).to(device)
   if device.startswith('npu'):
    import torch_npu
    obj['sc']=torch_npu.npu_trans_quant_param(obj['sw'])
   self.c.append(obj);return obj
  self.stem=folded('conv1','bn1');self.blocks=[]
  for stage in [1,2,3]:
   for i in range(3):
    p=f'layer{stage}.{i}';stride=2 if stage>1 and i==0 else 1
    c1=folded(p+'.conv1',p+'.bn1',stride);c2=folded(p+'.conv2',p+'.bn2')
    skip=folded(p+'.downsample.0',p+'.downsample.1',stride) if stride==2 else None
    self.blocks.append((c1,c2,skip))
  self.fw=state['fc.weight'].to(device);self.fb=state['fc.bias'].to(device)
 def fp(self,c,x):return F.conv2d(x,c['w'],c['b'],stride=c['stride'],padding=c['padding'])
 def codes(self,x,scale=None):
  sx=x.abs().amax((1,2,3),keepdim=True).clamp_min(1e-12)/127 if scale is None else scale
  return (x/sx).round().clamp(-127,127).to(torch.int8),sx
 def quant(self,c,x):
  q,sx=self.codes(x)
  if self.device.startswith('npu'):
   import torch_npu
   z=torch_npu.npu_quant_conv2d(q,c['qw'],c['sc'],[c['stride']]*2,[c['padding']]*2,[1,1],1,0,'rint',torch.float16).float()*sx
  else:z=F.conv2d(q.float(),c['qw'].float(),stride=c['stride'],padding=c['padding'])*c['sw'][None,:,None,None]*sx
  return z+c['b'][None,:,None,None]
 def target(self,c,x,s,precision='fp32',merged=True):
  # s divides output; stride-aware merged support implements the exact finite-sum identity.
  q=c['stride'];dtype=torch.float16 if precision=='fp16' else torch.float32
  oh=(x.shape[-2]+2*c['padding']-c['w'].shape[-2])//q+1
  ow=(x.shape[-1]+2*c['padding']-c['w'].shape[-1])//q+1
  if s==oh and s==ow:
   # Nine input summaries for a 3x3 kernel; never materialize the full FP layer.
   xp=F.pad(x.to(dtype),(c['padding'],)*4);kh,kw=c['w'].shape[-2:]
   v=torch.stack([xp[:,:,i:i+q*oh:q,j:j+q*ow:q].mean((-2,-1)) for i in range(kh) for j in range(kw)],-1)
   return (F.linear(v.flatten(1),c['w'].to(dtype).flatten(1),c['b'].to(dtype)))[:,:,None,None].float()
  if merged:
   key=(c['name'],s,precision)
   if key not in self.merged:self.merged[key]=expanded(c['w'],s,q).to(dtype)
   t=F.conv2d(x.to(dtype),self.merged[key],c['b'].to(dtype),stride=q*s,padding=c['padding'])
  else:
   assert q==1
   u=F.avg_pool2d(F.pad(x.to(dtype),(c['padding'],)*4),s,1)
   t=F.conv2d(u,c['w'].to(dtype),c['b'].to(dtype),stride=s)
  return t.float()
 def correction(self,c,x,z,mode='block',s=2,precision='fp32'):
  H,W=z.shape[-2:];s=H if s=='global' else s;hh=H//s*s;ww=W//s*s
  t=self.target(c,x,s,precision);t=t[...,:hh//s,:ww//s];delta=t-F.avg_pool2d(z[...,:hh,:ww],s);base=lift(delta,s);val=z.clone();val[...,:hh,:ww]+=base
  if mode=='smooth':
   v=F.interpolate(delta,size=(hh,ww),mode='bilinear',align_corners=False);val[...,:hh,:ww]+=v-lift(F.avg_pool2d(v,s),s)
  elif mode=='learned':
   assert s==2
   h=self.predict_completion(c,delta);val[...,:hh,:ww]+=h
  return val,delta
 def feature(self,d):
  p=F.pad(d,(1,1,1,1));H,W=d.shape[-2:]
  return torch.stack([p[:,:,i:i+H,j:j+W] for i in range(3) for j in range(3)],2).permute(1,2,0,3,4).flatten(2)
 def predict_completion(self,c,d):
  b=self.beta[c['name']];C=d.shape[1];N,H,W=d.shape[0],d.shape[-2],d.shape[-1]
  v=torch.bmm(b,self.feature(d)).reshape(C,4,N,H,W).permute(2,0,1,3,4);v=v-v.mean(2,keepdim=True)
  return v.reshape(N,C,2,2,H,W).permute(0,1,4,2,5,3).reshape(N,C,2*H,2*W)
 def op(self,c,x,mode,s=2,precision='fp32',fits=None):
  if mode in ['fp32','amp']:return self.fp(c,x)
  z=self.quant(c,x.float())
  if mode=='none':return z.half().float()
  if mode=='fit':
   cb,d=self.correction(c,x,z);y=self.fp(c,x);N,C,H,W=y.shape
   Y=(y-cb).reshape(N,C,H//2,2,W//2,2).permute(1,3,5,0,2,4).reshape(C,4,-1)
   X=self.feature(d);xx=arr(torch.bmm(X,X.transpose(1,2)));xy=arr(torch.bmm(X,Y.transpose(1,2)))
   if c['name'] not in fits:fits[c['name']]=[xx,xy,X.shape[-1]]
   else:fits[c['name']][0]+=xx;fits[c['name']][1]+=xy;fits[c['name']][2]+=X.shape[-1]
   return z.half().float()
  v,_=self.correction(c,x,z,mode,s,precision);return v.half().float()
 def forward(self,x,mode='fp32',s=2,precision='fp32',capture=None,fits=None):
  # Entry, projection shortcuts, pooling, residual adds and FC remain floating point.
  with torch.autocast('npu',dtype=torch.float16,enabled=mode=='amp') if self.device.startswith('npu') else torch.autocast('cpu',enabled=False):
   v=F.relu(self.fp(self.stem,x))
   for bi,(c1,c2,skipc) in enumerate(self.blocks):
    skip=v if skipc is None else self.fp(skipc,v)
    h=F.relu(self.op(c1,v,mode,s,precision,fits))
    if capture==bi:return h.detach().clone(),skip.detach().clone(),c2
    v=F.relu(self.op(c2,h,mode,s,precision,fits)+skip)
   return F.linear(v.mean((2,3)),self.fw,self.fb).float()
 def suffix(self,v,skip,bi,mode='fp32'):
  v=F.relu(v+skip)
  for c1,c2,sc in self.blocks[bi+1:]:
   skip=v if sc is None else self.fp(sc,v)
   v=F.relu(self.op(c2,F.relu(self.op(c1,v,mode)),mode)+skip)
  return F.linear(v.mean((2,3)),self.fw,self.fb).float()
 def fit_finish(self,fits):
  for n,(xx,xy,count) in fits.items():
   cv=xx/count;xy=xy/count;ridge=np.trace(cv,axis1=1,axis2=2)/9*1e-3+1e-12
   b=np.linalg.solve(cv+ridge[:,None,None]*np.eye(9),xy).transpose(0,2,1);b-=b.mean(1,keepdims=True);self.beta[n]=torch.tensor(b,dtype=torch.float32,device=self.device)
 def arrays(self):return {n:arr(b) for n,b in self.beta.items()}
def load_engine(device='npu:0',assets=ASSETS):
 state=torch.load(assets/'resnet20.pt',map_location='cpu',weights_only=True);return Engine(state,device),state
