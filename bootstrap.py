"""Obtain pinned upstream code and verify the public pretrained checkpoint."""
import argparse,hashlib,json,subprocess,urllib.request
from pathlib import Path

FULL='06c3181d9a0f7df3e30edbb1c355b6fec8b5a3ad'
CORE='2b4702beef0b823338e26713abc1688b03f6043b'
REV='afba5b47478f0c25c6a5a5755ac30738925d73ad'
NAME='checkpoints.20M.a4.last.ckpt'
SHA='3c8b571328112988fd7186953a2f3155f0e22301a9c8d7e24f520881251a5ef2'

def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',default='upstream');p.add_argument('--weights-dir',default='weights')
    a=p.parse_args();up=Path(a.upstream)
    if not up.exists():
        subprocess.run(['git','clone','https://github.com/EveNet-HEP/EveNet-Full.git',str(up)],check=True)
        git(up,'checkout',FULL)
        git(up,'config','submodule.evenet.url','https://github.com/EveNet-HEP/Core.git')
        git(up,'submodule','update','--init')
    # Existing trees are checked, never reset or cleaned.
    if git(up,'rev-parse','HEAD')!=FULL or git(up/'evenet','rev-parse','HEAD')!=CORE:
        raise RuntimeError('Upstream revisions differ from the tested pins; use a new destination.')
    for tree in [up,up/'evenet']:
        if git(tree,'status','--porcelain','--untracked-files=no'):
            raise RuntimeError(f'Tracked upstream modifications found: {tree}')
    weights=Path(a.weights_dir);weights.mkdir(parents=True,exist_ok=True);target=weights/NAME
    if not target.exists():
        temporary=target.with_suffix('.download')
        url=f'https://huggingface.co/Avencast/EveNet/resolve/{REV}/{NAME}'
        urllib.request.urlretrieve(url,temporary)
        with temporary.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=SHA: raise RuntimeError('Downloaded checkpoint checksum mismatch')
        temporary.replace(target)
    with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=SHA:raise RuntimeError('Existing checkpoint checksum mismatch; no weights loaded')
    print(json.dumps({'upstream_commit':FULL,'core_commit':CORE,'checkpoint_sha256':digest,'checkpoint':str(target)},indent=2))

if __name__=='__main__':main()
