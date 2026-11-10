"""Convert selected all-hadronic AK4 ntuples to raw EveNet NPZ shards.

Requires real MET branches and an explicit lepton/tau-veto declaration. Does
not substitute zero for missing measurements. Does not run analysis selections.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
from model import OBJECT_FEATURES,GLOBAL_FEATURES

def split_of(key):
    value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')/2**64
    return 'train' if value<.7 else 'val' if value<.85 else 'test'

def invariant(p4):
    q=p4.sum(axis=0)
    return np.sqrt(max(q[3]**2-np.dot(q[:3],q[:3]),0.))

def build_event(px,py,pz,energy,btag,met,met_phi,wp):
    arrays=[np.asarray(v,dtype=np.float64) for v in [px,py,pz,energy,btag]]
    if len({len(v) for v in arrays})!=1:raise ValueError('Unequal AK4 branch lengths')
    if not all(np.isfinite(v).all() for v in arrays):raise ValueError('Nonfinite jet')
    px,py,pz,e,b=arrays
    pt=np.hypot(px,py)
    if len(pt)==0 or (pt<=0).any() or (e<=0).any():raise ValueError('Empty/invalid selected jets')
    if not np.isfinite([met,met_phi]).all() or met<0:raise ValueError('Invalid MET')
    # This ordering only selects/pads objects; it never assigns jets to chi candidates.
    idx=np.argsort(-pt,kind='stable')[:18]
    p4=np.stack([px[idx],py[idx],pz[idx],e[idx]],axis=1)
    bit=(b[idx]>wp).astype(np.float32)
    x=np.zeros((18,7),dtype=np.float32);mask=np.zeros(18,dtype=bool)
    x[:len(idx),:5]=np.stack([e[idx],pt[idx],np.arcsinh(pz[idx]/pt[idx]),np.arctan2(py[idx],px[idx]),bit],axis=1)
    mask[:len(idx)]=True
    # Match the public CMS GridStudy global-feature convention, all-hadronic limit.
    # nJet counts selected jets before truncation; sums use retained objects.
    g=np.array([met,np.arctan2(np.sin(met_phi),np.cos(met_phi)),0,bit.sum(),len(pt),
                pt[idx].sum(),0,invariant(p4),0,invariant(p4[bit==1])],dtype=np.float32)
    return x,g,mask,len(pt)>18

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--output',required=True)
    a=p.parse_args();samples=json.loads(Path(a.manifest).read_text());out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('*.npz')):raise FileExistsError('Use an empty output directory')
    seen=set();report=[]
    for file_index,s in enumerate(samples):
        if s.get('selected_zero_leptons_and_taus') is not True:
            raise ValueError('This jet-only converter requires a documented lepton/tau veto; use a general object converter otherwise.')
        if s.get('label') not in [0,1,None]:raise ValueError('Use 0/1 for MC training labels or null for unlabeled inference data')
        if s['label']==1 and (min(s['m_chi_gev'],s['m_suu_gev'])<=0):raise ValueError('Positive signal masses required')
        if not np.isfinite(s['btag_wp']):raise ValueError('Explicit btag working point required')
        b={'px':'AK4_px','py':'AK4_py','pz':'AK4_pz','energy':'AK4_E','btag':'AK4_DeepJet_disc',
           'run':'run','lumi':'luminosityBlock','event':'event',**s.get('branches',{})}
        if 'met' not in b or 'met_phi' not in b:raise ValueError('Map real MET pt and phi branches; original event tree has neither.')
        names=list(dict.fromkeys(b.values()))
        with uproot.open(s['file']) as f:
            tree=f[s['tree']]
            missing=set(names)-set(tree.keys())
            if missing:raise ValueError(f'Missing branches: {sorted(missing)}')
            for chunk_index,arr in enumerate(tree.iterate(names,step_size='100 MB',library='ak')):
                buckets={k:[] for k in ['train','val','test']}
                for i in range(len(arr)):
                    # Same namespace for nominal/systematic copies; dataset-wide uniqueness
                    # must be established by the producer (MC IDs can repeat across jobs).
                    key=':'.join([s['event_namespace']]+[str(int(arr[b[k]][i])) for k in ['run','lumi','event']])
                    if key in seen:raise ValueError(f'Duplicate event key {key}; resolve producer event identity')
                    seen.add(key)
                    v=[ak.to_numpy(arr[b[k]][i]) for k in ['px','py','pz','energy','btag']]
                    x,g,m,truncated=build_event(*v,float(arr[b['met']][i]),float(arr[b['met_phi']][i]),s['btag_wp'])
                    w=float(arr[b['train_weight']][i]) if 'train_weight' in b else float(s.get('train_weight',1.))
                    if not np.isfinite(w) or w<0:raise ValueError('Explicit nonnegative training weights required')
                    target=[s['m_chi_gev'],s['m_suu_gev']] if s['label']==1 else [np.nan,np.nan]
                    buckets[split_of(key)].append((x,g,m,key,-1 if s.get('label') is None else s['label'],target,w,truncated))
                for split,rows in buckets.items():
                    if not rows:continue
                    path=out/f'{split}_{file_index:04d}_{chunk_index:04d}.npz'
                    columns=list(zip(*rows))
                    data={k:np.asarray(v) for k,v in zip(['x','globals','mask','event_key','label','mass_gev','train_weight','truncated'],columns)}
                    np.savez_compressed(path,**data,schema=np.array('evenet-suu-raw-v1'),
                                        object_features=np.array(OBJECT_FEATURES),global_features=np.array(GLOBAL_FEATURES))
                    report.append({'file':path.name,'events':len(rows),'truncated':int(sum(columns[-1]))})
    (out/'conversion.json').write_text(json.dumps({'manifest':samples,'shards':report},indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
