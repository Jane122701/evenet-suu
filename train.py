"""Fine tune or predict from prepared NPZ shards; see README.md."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from model import SuuModel,loss_parts,joint_loss,OBJECT_FEATURES,GLOBAL_FEATURES

def read_data(paths, training=True):
    fields=['x','globals','mask','event_key']
    if training: fields += ['label','mass_gev','train_weight']
    chunks=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as f:
            if f['schema'].item()!='evenet-suu-raw-v1': raise ValueError(f'{path}: wrong schema')
            if f['object_features'].tolist()!=OBJECT_FEATURES or f['global_features'].tolist()!=GLOBAL_FEATURES:
                raise ValueError(f'{path}: incorrect feature order')
            chunks.append({k:f[k] for k in fields})
    d={k:np.concatenate([c[k] for c in chunks]) for k in fields}
    n=len(d['x'])
    if n==0: raise ValueError('Empty dataset')
    if any(len(v)!=n for v in d.values()): raise ValueError('Inconsistent lengths')
    x,g,m=d['x'],d['globals'],d['mask']
    if x.shape!=(n,18,7) or g.shape!=(n,10) or m.shape!=(n,18): raise ValueError('Expected x[N,18,7], globals[N,10], mask[N,18]')
    if not np.isin(m,[0,1]).all() or not (m.sum(1)>0).all(): raise ValueError('Invalid/empty object masks')
    if not np.isfinite(x).all() or not np.isfinite(g).all(): raise ValueError('Nonfinite inputs')
    if (x[...,:2]<0).any() or (g[:,[0,5,6,7,8,9]]<0).any(): raise ValueError('Negative log1p input')
    if (abs(x[...,3])>np.pi+1e-5).any() or (abs(g[:,1])>np.pi+1e-5).any(): raise ValueError('Phi must be in [-pi,pi]')
    if not np.isin(x[...,4],[0,1,2]).all(): raise ValueError('btag must be 0/1 (2 reserved for tau), not a continuous discriminator')
    if not np.isin(x[...,5],[0,1]).all(): raise ValueError('isLepton must be binary')
    if len(set(d['event_key'].tolist()))!=n: raise ValueError('Duplicate event keys within dataset')
    if training:
        y,t,w=d['label'],d['mass_gev'],d['train_weight']
        if y.shape!=(n,) or t.shape!=(n,2) or w.shape!=(n,): raise ValueError('Bad target/weight shape')
        if not np.isin(y,[0,1]).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0: raise ValueError('Invalid labels or training weights')
        if not np.isfinite(t[y==1]).all() or (t[y==1]<=0).any(): raise ValueError('Invalid signal mass target')
    return d

def loader(d,batch,shuffle=False,training=True):
    keys=['x','globals','mask']+(['label','mass_gev','train_weight'] if training else [])
    arrays=[torch.from_numpy(np.asarray(d[k]).astype(bool if k=='mask' else np.float32)) for k in keys]
    return DataLoader(TensorDataset(*arrays),batch_size=batch,shuffle=shuffle)

def validate_training_pair(tr,va):
    if set(tr['event_key'].tolist()) & set(va['event_key'].tolist()):
        raise ValueError('Train/validation event overlap')
    for name,d in [('train',tr),('val',va)]:
        if set(d['label'].tolist())!={0.,1.}:
            raise ValueError(f'{name} needs signal AND background')
        if any(d['train_weight'][d['label']==y].sum()<=0 for y in [0,1]):
            raise ValueError(f'{name}: zero class weight')

def epoch(model,dl,device,opt=None,mass_weight=1.,frozen=False):
    model.train(opt is not None)
    if frozen: model.backbone.eval()  # disable dropout in frozen feature extraction
    totals=np.zeros(4,dtype=np.float64)
    with torch.set_grad_enabled(opt is not None):
        for batch in dl:
            x,g,mask,y,t,w=[v.to(device) for v in batch]
            logits,m=model(x,g,mask)
            parts=loss_parts(logits,m,y,t,w)
            loss=joint_loss(parts,mass_weight)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            if opt is not None:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
            totals += np.array([v.detach().cpu().item() for v in parts])
    c,cn,m,mn=totals
    return {'classification':c/max(cn,1e-12),'mass':m/max(mn,1e-12),
            'total':c/max(cn,1e-12)+mass_weight*m/max(mn,1e-12)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','predict'])
    p.add_argument('--upstream',required=True)
    p.add_argument('--pretrained')
    p.add_argument('--checkpoint')
    p.add_argument('--train',nargs='+');p.add_argument('--val',nargs='+');p.add_argument('--input',nargs='+')
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--freeze-epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--backbone-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--mass-weight',type=float,default=1.)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cpu-threads',type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(a.cpu_threads); torch.manual_seed(a.seed);np.random.seed(a.seed)
    model=SuuModel(a.upstream)
    if a.mode=='predict':
        if not a.checkpoint or not a.input: p.error('predict requires --checkpoint and --input')
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(ck['model'],strict=True);model.to(a.device).eval()
        d=read_data(a.input,training=False);scores=[];masses=[]
        with torch.inference_mode():
            for x,g,m in loader(d,a.batch_size,training=False):
                l,t=model(x.to(a.device),g.to(a.device),m.to(a.device))
                pred=1000.*torch.exp(t)
                if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite mass prediction')
                scores.append(l.sigmoid().cpu().numpy());masses.append(pred.cpu().numpy())
        np.savez_compressed(a.output,event_key=d['event_key'],event_score=np.concatenate(scores),
                            mass_gev=np.concatenate(masses),mass_names=np.array(['m_chi','m_suu']))
        return
    if not a.pretrained or not a.train or not a.val: p.error('train requires --pretrained, --train, --val')
    if a.epochs<1 or not 0<=a.freeze_epochs<=a.epochs or a.mass_weight<0: p.error('Invalid epoch/loss options')
    tr,va=read_data(a.train),read_data(a.val)
    validate_training_pair(tr,va)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists(): raise FileExistsError('Choose a new output directory; best.pt already exists')
    audit=model.load_pretrained(a.pretrained)
    (out/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    model.to(a.device)
    opt=torch.optim.AdamW([
        {'params':model.backbone.parameters(),'lr':a.backbone_lr},
        {'params':list(model.event_head.parameters())+list(model.mass_head.parameters()),'lr':a.head_lr}],weight_decay=.01)
    td,vd=loader(tr,a.batch_size,True),loader(va,a.batch_size)
    best=float('inf');history=[]
    for i in range(a.epochs):
        frozen=i<a.freeze_epochs
        model.backbone.requires_grad_(not frozen)
        metrics={'epoch':i+1,'frozen_backbone':frozen,
                 'train':epoch(model,td,a.device,opt,a.mass_weight,frozen),
                 'val':epoch(model,vd,a.device,mass_weight=a.mass_weight)}
        history.append(metrics);print(json.dumps(metrics),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if metrics['val']['total']<best:
            best=metrics['val']['total']
            torch.save({'model':model.state_dict(),'args':vars(a),'audit':audit,'epoch':i+1,
                        'object_features':OBJECT_FEATURES,'global_features':GLOBAL_FEATURES},out/'best.pt')

if __name__=='__main__':main()
