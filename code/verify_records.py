from pathlib import Path
import json,hashlib,math
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'MANIFEST.json').read_text())
for name,record in manifest.items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==record['sha256'],name
load=lambda n:json.loads((R/'data'/n).read_text())
rows=load('mechanism_all_cases.json');assert len(rows)==720
assert len(load('adverse_and_false_alarm_cases.json'))==31
margins={r['case']:r for r in load('margin_cases.json')}
for r in rows:
 assert math.isclose(r['K']+r['C'],r['L'],rel_tol=1e-10,abs_tol=1e-15)
 m=margins[r['case']]
 for k,v in [('abs_linear_over_T',abs(r['L'])/r['T']),('abs_remainder_over_T',abs(r['remainder'])/r['T']),('abs_remainder_over_linear',abs(r['remainder'])/abs(r['L']))]:
  assert math.isclose(m[k],v,rel_tol=1e-12), (r['case'],k)
 assert math.isclose(m['T_hat'],r['L']+2*r['K']*r['v_over_u_cast']**2,rel_tol=1e-12)
def quantile(a,p):
 a=sorted(a);h=(len(a)-1)*p;i=int(h);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(h-i)
for g in load('margin_groups.json'):
 for k,qs in g['quartiles'].items():
  values=[r[k] for r in margins.values() if r['group']==g['group']]
  assert all(math.isclose(v,quantile(values,p),rel_tol=1e-12) for v,p in zip(qs,[.25,.5,.75]))
assert [x['n'] for x in load('margin_groups.json')]==[689,24,2,5]
assert [(x['fp64_positive'],x['fp64_negative'],x['fp64_unresolved']) for x in load('ce_summary.json') if x['block']=='all']==[(375,345,0),(369,351,0)]
a=load('abstention_curve.json');assert [x['threshold'] for x in a]==[0,.01,.025,.05,.1,.2,.5,.9]
assert all(x['correct']+x['wrong']+x['unresolved']==x['retained'] for x in a)
for x in a:
 keep=[m for m in margins.values() if m['linear_margin_fraction']>=x['threshold']]
 assert len(keep)==x['retained']
 assert sum(m['actual_sign']<0 for m in keep)==x['actual_negative']
 assert sum(m['actual_sign']!=m['predicted_sign'] for m in keep)==x['wrong']
print('PASS: file hashes, 720 records, 31 adverse/false alarms, CE counts and threshold grid. No new inference.')
