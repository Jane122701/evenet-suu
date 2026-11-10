"""Integration checks with the real pretrained backbone and synthetic events.

These establish software behavior, not physics performance.
"""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
import awkward as ak
import uproot
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import SuuModel,loss_parts,joint_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--pretrained',required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(3)
    m=SuuModel(a.upstream);audit=m.load_pretrained(a.pretrained)
    assert audit['loaded_backbone_tensors']==audit['expected_backbone_tensors']
    # Signal-only regression: background NaNs and arbitrary values give identical loss.
    logits=torch.tensor([.2,-.1,.6],requires_grad=True)
    pred=torch.tensor([[.1,.2],[.3,.4],[.5,.6]],requires_grad=True)
    y=torch.tensor([0.,1.,0.]);target=torch.tensor([[float('nan')]*2,[1000.,3000.],[float('nan')]*2])
    parts=loss_parts(logits,pred,y,target)
    parts[2].backward(retain_graph=True)
    assert torch.equal(pred.grad[[0,2]],torch.zeros(2,2)) and pred.grad[1].abs().sum()>0
    alt=target.clone();alt[[0,2]]=99999.
    assert torch.equal(parts[2],loss_parts(logits,pred,y,alt)[2])
    bg=loss_parts(logits,pred,torch.zeros(3),torch.full((3,2),float('nan')))
    assert bg[2].item()==0 and torch.isfinite(joint_loss(bg))
    # Missing pretrained tensors must be fatal.
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp)
        broken={'state_dict':{'model.'+k:v for k,v in m.backbone.state_dict().items()}}
        broken['state_dict'].pop(next(iter(broken['state_dict'])))
        torch.save(broken,t/'broken.pt')
        try:m.load_pretrained(t/'broken.pt')
        except RuntimeError:pass
        else:raise AssertionError('Missing backbone tensor accepted')
        rng=np.random.default_rng(4);manifest=[]
        for label in [0,1]:
            n=40;nj=6;px=rng.normal(100,80,(n,nj));py=rng.normal(0,80,(n,nj));pz=rng.normal(0,100,(n,nj))
            energy=np.sqrt(px**2+py**2+pz**2+15**2)
            path=t/f'sample{label}.root'
            with uproot.recreate(path) as f:
                f['Events']={'AK4_px':ak.Array(px.tolist()),'AK4_py':ak.Array(py.tolist()),
                    'AK4_pz':ak.Array(pz.tolist()),'AK4_E':ak.Array(energy.tolist()),
                    'AK4_DeepJet_disc':ak.Array(rng.uniform(0,1,(n,nj)).tolist()),
                    'run':np.ones(n,dtype=np.uint32),'luminosityBlock':np.ones(n,dtype=np.uint32),
                    'event':np.arange(n,dtype=np.uint64),'MET_pt':np.full(n,30.),'MET_phi':np.zeros(n)}
            s={'file':str(path),'tree':'Events','event_namespace':f'toy{label}','label':label,
               'btag_wp':.2783,'selected_zero_leptons_and_taus':True,
               'branches':{'met':'MET_pt','met_phi':'MET_phi'}}
            if label:s.update(m_chi_gev=1000,m_suu_gev=3000)
            manifest.append(s)
        (t/'manifest.json').write_text(json.dumps(manifest))
        root=Path(__file__).resolve().parents[1]
        def run(*args):
            subprocess.run([sys.executable,*map(str,args)],check=True,cwd=root)
        run(root/'prepare_root.py',t/'manifest.json','--output',t/'data')
        tr=sorted((t/'data').glob('train*.npz'));va=sorted((t/'data').glob('val*.npz'));te=sorted((t/'data').glob('test*.npz'))
        run(root/'train.py','train','--upstream',Path(a.upstream).resolve(),'--pretrained',Path(a.pretrained).resolve(),
            '--train',*tr,'--val',*va,'--output',t/'run','--epochs',2,'--freeze-epochs',1,'--batch-size',32,'--cpu-threads',2,'--device','cpu')
        ck=torch.load(t/'run/best.pt',map_location='cpu',weights_only=True)
        m.load_state_dict(ck['model']);m.eval()
        # Check real backbone backward pass explicitly, independent of best epoch.
        d=np.load(tr[0]);x=torch.tensor(d['x'][:3]).float();g=torch.tensor(d['globals'][:3]).float();mask=torch.tensor(d['mask'][:3])
        l,r=m(x,g,mask);assert l.shape==(3,) and r.shape==(3,2)
        m.zero_grad();(l.sum()+r.sum()).backward()
        assert m.backbone.PET.feature_embedding[0].weight.grad.norm()>0
        run(root/'train.py','predict','--upstream',Path(a.upstream).resolve(),'--checkpoint',t/'run/best.pt',
            '--input',*te,'--output',t/'predictions.npz','--device','cpu','--cpu-threads',2)
        with np.load(t/'predictions.npz') as result:
            assert np.isfinite(result['mass_gev']).all() and (result['mass_gev']>0).all()
            assert ((result['event_score']>=0)&(result['event_score']<=1)).all()
            assert len(result['event_score'])==sum(len(np.load(f)['label']) for f in te)
        # Collision-data conversion/inference does not need truth targets.
        manifest[0]['label']=None
        (t/'unlabeled.json').write_text(json.dumps(manifest[:1]))
        run(root/'prepare_root.py',t/'unlabeled.json','--output',t/'unlabeled')
        unlabeled=sorted((t/'unlabeled').glob('*.npz'))
        run(root/'train.py','predict','--upstream',Path(a.upstream).resolve(),'--checkpoint',t/'run/best.pt',
            '--input',*unlabeled,'--output',t/'unlabeled_predictions.npz','--device','cpu','--cpu-threads',2)
        assert len(np.load(t/'unlabeled_predictions.npz')['event_score'])==40
        print('PASS: strict pretrained loading, signal-only/empty-signal mass loss, ROOT conversion, frozen and unfrozen training, validation, checkpoint reload, all-event inference.')

if __name__=='__main__':main()
