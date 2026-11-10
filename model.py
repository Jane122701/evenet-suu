"""EveNet-Full deterministic backbone + new Suu task heads.

Uses unmodified upstream Core modules, pinned by setup.sh. Inputs are raw GeV
features; log1p and released checkpoint normalization happen here exactly once.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
import yaml

CORE_COMMIT = '2b4702beef0b823338e26713abc1688b03f6043b'
FULL_COMMIT = '06c3181d9a0f7df3e30edbb1c355b6fec8b5a3ad'
HF_REVISION = 'afba5b47478f0c25c6a5a5755ac30738925d73ad'
OBJECT_FEATURES = ['energy','pt','eta','phi','btag','isLepton','charge']
GLOBAL_FEATURES = ['met','met_phi','nLepton','nbJet','nJet','HT','HT_lep','M_all','M_leps','M_bjets']

def setup_import(upstream):
    p = Path(upstream).resolve()
    sys.path.insert(0, str(p))
    import evenet.network.body.embedding as embedding
    if not Path(embedding.__file__).resolve().is_relative_to(p):
        raise RuntimeError('A different evenet installation was imported. Use a fresh process.')
    return p

class Backbone(nn.Module):
    def __init__(self, upstream):
        super().__init__()
        p = setup_import(upstream)
        from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
        from evenet.network.body.object_encoder import ObjectEncoder
        from evenet.network.body.normalizer import Normalizer
        cfg = yaml.safe_load((p/'share/network/network-20M.yaml').read_text())['Body']
        g, t, o = cfg['GlobalEmbedding'], cfg['PET'], cfg['ObjectEncoder']
        self.sequential_normalizer = Normalizer(torch.zeros(7),torch.ones(7),torch.tensor([1,1,1,1,0,0,0],dtype=torch.bool),inv_cdf_index=[3])
        self.global_normalizer = Normalizer(torch.zeros(10),torch.ones(10),torch.tensor([1,1,0,0,0,1,1,1,1,1],dtype=torch.bool))
        self.GlobalEmbedding = GlobalVectorEmbedding(
            linear_block_type=g['linear_block_type'],input_dim=10,
            hidden_dim_scale=g['transformer_dim_scale'],initial_embedding_dim=g['initial_embedding_dim'],
            final_embedding_dim=g['hidden_dim'],normalization_type=g['normalization'],
            activation_type=g['linear_activation'],dropout=g['dropout'],
            skip_connection=g['skip_connection'],num_embedding_layers=g['num_embedding_layers'])
        self.PET = PETBody(num_feat=7,num_keep=t['num_feature_keep'],feature_drop=0.0,
            projection_dim=t['hidden_dim'],local=t['enable_local_embedding'],K=t['local_Krank'],
            num_local=t['num_local_layer'],num_layers=t['num_layers'],num_heads=t['num_heads'],
            drop_probability=t['drop_probability'],talking_head=t['talking_head'],
            layer_scale=t['layer_scale'],layer_scale_init=t['layer_scale_init'],dropout=t['dropout'],mode=t['mode'])
        self.ObjectEncoder = ObjectEncoder(input_dim=t['hidden_dim'],hidden_dim=o['hidden_dim'],
            output_dim=o['hidden_dim'],position_embedding_dim=o['position_embedding_dim'],
            num_heads=o['num_attention_heads'],transformer_dim_scale=o['transformer_dim_scale'],
            num_linear_layers=o['num_embedding_layers'],num_encoder_layers=o['num_encoder_layers'],
            dropout=o['dropout'],skip_connection=o['skip_connection'],encoder_skip_connection=o['encoder_skip_connection'],conditioned=False)
        self.output_dim = o['hidden_dim']
        self.local_indices = t['local_point_index']

    def forward(self, x, globals_, mask):
        # validate_data performs full checks once before training; keep hot path small.
        x = x.clone()
        globals_ = globals_.clone()
        x[..., :2] = torch.log1p(x[..., :2])
        globals_[..., [0,5,6,7,8,9]] = torch.log1p(globals_[..., [0,5,6,7,8,9]])
        m = mask.bool().unsqueeze(-1)
        gm = torch.ones((x.shape[0],1,1),device=x.device,dtype=torch.bool)
        x = self.sequential_normalizer(x,m)
        g = self.global_normalizer(globals_.unsqueeze(1),gm)
        g = self.GlobalEmbedding(g,gm)
        z = self.PET(input_features=x,input_points=x[...,self.local_indices],mask=m,
                     time=x.new_zeros(x.shape[0]),time_masking=torch.zeros_like(m,dtype=x.dtype))
        return self.ObjectEncoder(z,m,g,gm)[2]

class SuuModel(nn.Module):
    def __init__(self, upstream):
        super().__init__()
        self.backbone = Backbone(upstream)
        d = self.backbone.output_dim
        self.event_head = nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,1))
        self.mass_head = nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,2))

    def forward(self,x,globals_,mask):
        h = self.backbone(x,globals_,mask)
        return self.event_head(h).squeeze(-1), self.mass_head(h)

    def load_pretrained(self,path,weights='state_dict'):
        # Never falls back to arbitrary pickle execution or silently skips a backbone layer.
        ckpt = torch.load(path,map_location='cpu',weights_only=True)
        if weights not in ckpt or not isinstance(ckpt[weights],dict):
            raise ValueError(f'Missing checkpoint mapping: {weights}')
        source = ckpt[weights]
        expected = self.backbone.state_dict()
        chosen, missing, bad, keys = {}, [], [], []
        for name, target in expected.items():
            candidates = [k for k in source if k == name or k.endswith('.'+name)]
            if len(candidates) != 1:
                missing.append({'name':name,'candidates':candidates}); continue
            key = candidates[0]
            if not isinstance(source[key],torch.Tensor) or source[key].shape != target.shape:
                bad.append(name); continue
            chosen[name] = source[key]; keys.append(key)
        if missing or bad:
            raise RuntimeError(json.dumps({'missing_or_ambiguous':missing,'shape_mismatch':bad},indent=2))
        self.backbone.load_state_dict(chosen,strict=True)
        with open(path,'rb') as f:
            digest = hashlib.file_digest(f,'sha256').hexdigest()
        return {'checkpoint_sha256':digest,'weights':weights,'loaded_backbone_tensors':len(chosen),
                'expected_backbone_tensors':len(expected),'loaded_backbone_elements':sum(v.numel() for v in chosen.values()),
                'ignored_pretraining_tensors':len(source)-len(keys),'core_commit':CORE_COMMIT,
                'full_commit':FULL_COMMIT,'hf_revision':HF_REVISION}

def loss_parts(logits, log_mass_tev, labels, mass_gev, weights=None):
    """Return differentiable numerator/denominator pairs for exact epoch averages.

    Background mass may be NaN: select signal BEFORE log or regression loss.
    Training weights must be nonnegative; retain signed physics weights separately.
    """
    w = torch.ones_like(logits) if weights is None else weights
    if not torch.isfinite(w).all() or (w<0).any():
        raise ValueError('Training weights must be finite and nonnegative.')
    if not ((labels==0)|(labels==1)).all():
        raise ValueError('Classification labels must be 0 or 1.')
    cls_num = (F.binary_cross_entropy_with_logits(logits,labels.float(),reduction='none')*w).sum()
    sig = labels==1
    if sig.any():
        target = mass_gev[sig]
        if not torch.isfinite(target).all() or (target<=0).any():
            raise ValueError('Both signal mass targets must be finite and positive.')
        per_signal = F.smooth_l1_loss(log_mass_tev[sig],torch.log(target/1000.),reduction='none').mean(-1)
        mass_num = (per_signal*w[sig]).sum()
    else:
        mass_num = log_mass_tev.sum()*0.0
    return cls_num,w.sum(),mass_num,w[sig].sum()

def joint_loss(parts,mass_weight=1.):
    c,cn,m,mn=parts
    return c/cn.clamp_min(1e-12)+mass_weight*m/mn.clamp_min(1e-12)
