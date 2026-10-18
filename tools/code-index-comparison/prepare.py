#!/usr/bin/env python3
import json, os, secrets, subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
REVISION='cfb58ee2f0927c975f3fe7e19dd60e1cbf646fdf'
def prepare():
    root=BASE/'sources'/'po'
    if not root.exists():
        root.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout','--filter=blob:none','https://github.com/this-rs/project-orchestrator.git',str(root)],check=True)
        subprocess.run(['git','-C',str(root),'checkout','--detach',REVISION],check=True)
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
    if revision!=REVISION or dirty:
        raise SystemExit('Source checkout differs from the pinned clean snapshot; refusing to replace it.')
    env=BASE/'.env'
    if not env.exists():
        with env.open('x') as f:
            f.write('NEO4J_PASSWORD='+secrets.token_hex(24)+'\nMEILI_MASTER_KEY='+secrets.token_hex(24)+'\n')
        os.chmod(env,0o600)
    (BASE/'results').mkdir(exist_ok=True)
    manifest={'source_revision':REVISION,'po_runtime':'v0.0.17','po_runtime_revision':'8a8f0b770beb10bf58075f8f89ad2e99c200ee4f','cbm_runtime':'v0.11.0',
              'note':'PO parser and resolver unchanged between runtime release and source snapshot; whole runtime revisions differ.',
              'embeddings':'PO disabled; comparison concerns structural indexing and keyword search',
              'execution_status':'prepared; not yet measured'}
    (BASE/'results'/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Pinned source and local configuration prepared.')
if __name__=='__main__': prepare()
