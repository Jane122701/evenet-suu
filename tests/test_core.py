"""Fast contract tests; no upstream download or pretrained weights required."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from model import loss_parts,joint_loss,OBJECT_FEATURES,GLOBAL_FEATURES
from prepare_root import build_event,split_of
from train import read_data,validate_training_pair

class LossTests(unittest.TestCase):
    def test_background_targets_and_gradients_excluded(self):
        z=torch.zeros(3,2,requires_grad=True);y=torch.tensor([0.,1.,0.])
        t=torch.tensor([[float('nan')]*2,[2000.,6000.],[float('nan')]*2])
        a=loss_parts(torch.zeros(3),z,y,t)
        t[[0,2]]=100000.
        b=loss_parts(torch.zeros(3),z,y,t)
        self.assertTrue(torch.equal(a[2],b[2]));a[2].backward()
        self.assertEqual(z.grad[[0,2]].abs().sum().item(),0.)
        self.assertGreater(z.grad[1].abs().sum().item(),0.)

    def test_background_only_batch(self):
        z=torch.zeros(2,2,requires_grad=True)
        parts=loss_parts(torch.zeros(2,requires_grad=True),z,torch.zeros(2),torch.full((2,2),float('nan')))
        self.assertEqual(parts[2].item(),0.);self.assertTrue(torch.isfinite(joint_loss(parts)))
        joint_loss(parts).backward();self.assertEqual(z.grad.abs().sum().item(),0.)

    def test_negative_training_weights_rejected(self):
        with self.assertRaises(ValueError):
            loss_parts(torch.zeros(1),torch.zeros(1,2),torch.zeros(1),torch.full((1,2),float('nan')),torch.tensor([-1.]))

    def test_invalid_signal_mass_rejected(self):
        with self.assertRaises(ValueError):
            loss_parts(torch.zeros(1),torch.zeros(1,2),torch.ones(1),torch.tensor([[0.,1000.]]))

    def test_weighted_epoch_sums(self):
        args=[torch.tensor([0.,1.,-1.]),torch.zeros(3,2),torch.tensor([0.,1.,1.]),
              torch.tensor([[float('nan')]*2,[1000.,3000.],[2000.,6000.]]),torch.tensor([1.,2.,3.])]
        full=loss_parts(*args)
        first=loss_parts(*[v[:1] for v in args]);rest=loss_parts(*[v[1:] for v in args])
        self.assertTrue(all(torch.allclose(x,a+b) for x,a,b in zip(full,first,rest)))

class DataTests(unittest.TestCase):
    def fixture(self):
        x,g,m,cut=build_event([100.,-100.],[0.,0.],[0.,0.],[110.,110.],[.9,.1],30.,0.,.2783)
        return {'x':x[None],'globals':g[None],'mask':m[None],'event_key':np.array(['sample:1:1:1']),
                'schema':np.array('evenet-suu-raw-v1'),'object_features':np.array(OBJECT_FEATURES),
                'global_features':np.array(GLOBAL_FEATURES)}

    def read_fixture(self,d,training=False):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'data.npz';np.savez(path,**d)
            return read_data([path],training=training)

    def test_conversion_physics_and_padding(self):
        d=self.fixture();g=d['globals'][0]
        self.assertAlmostEqual(g[7],220.)
        self.assertEqual(g[3],1);self.assertEqual(g[4],2);self.assertEqual(g[5],200.)
        self.assertTrue((d['x'][0,2:]==0).all());self.assertEqual(d['mask'].sum(),2)

    def test_inference_without_any_target_arrays(self):
        d=self.read_fixture(self.fixture());self.assertNotIn('label',d);self.assertNotIn('mass_gev',d)

    def test_continuous_btag_rejected(self):
        d=self.fixture();d['x'][0,0,4]=.6
        with self.assertRaises(ValueError):self.read_fixture(d)

    def test_wrong_feature_order_rejected(self):
        d=self.fixture();d['object_features']=d['object_features'][::-1]
        with self.assertRaises(ValueError):self.read_fixture(d)

    def test_duplicate_events_rejected(self):
        d=self.fixture()
        for k in ['x','globals','mask','event_key']:d[k]=np.concatenate([d[k],d[k]])
        with self.assertRaises(ValueError):self.read_fixture(d)

    def test_cross_split_overlap_rejected(self):
        d={'event_key':np.array(['a','b']),'label':np.array([0,1]),'train_weight':np.ones(2)}
        with self.assertRaisesRegex(ValueError,'overlap'):validate_training_pair(d,d)

    def test_split_stable(self):
        self.assertEqual(split_of('same:event'),split_of('same:event'))

    def test_zero_pt_jet_rejected(self):
        with self.assertRaises(ValueError):build_event([0.],[0.],[1.],[2.],[.1],10.,0.,.3)

if __name__=='__main__':unittest.main()
