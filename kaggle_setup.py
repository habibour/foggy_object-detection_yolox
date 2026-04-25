"""
kaggle_setup.py — Run this once in Kaggle to set everything up.
Usage in Kaggle cell: !python /kaggle/working/kaggle_setup.py
"""
import os, sys, shutil, subprocess
from pathlib import Path

WORK   = Path('/kaggle/working')
BASE   = Path('/kaggle/input/datasets/mdhabibourrahman/object-detection-dataset')
RTTS   = BASE / 'RTTS_Ready'   / 'RTTS_Ready'
VOCFOG = BASE / 'VOC_FOG_YOLO' / 'VOC_FOG_YOLO'

# ── 1. Clone / update code from GitHub ───────────────────────────────────────
REPO = WORK / 'repo'
if REPO.exists():
    subprocess.run(['git', '-C', str(REPO), 'pull'], check=True)
else:
    subprocess.run(['git', 'clone',
        'https://github.com/habibour/foggy_object-detection_yolox.git',
        str(REPO)], check=True)

shutil.copytree(str(REPO / 'said'), str(WORK / 'said'), dirs_exist_ok=True)
shutil.copy2(str(REPO / 'train.py'), str(WORK / 'train.py'))
ver = subprocess.check_output(['git','-C',str(REPO),'log','--oneline','-1']).decode().strip()
print(f'Code: {ver}')

# ── 2. Verify datasets ────────────────────────────────────────────────────────
assert RTTS.exists(),   f'RTTS not found: {RTTS}'
assert VOCFOG.exists(), f'VOC-FOG not found: {VOCFOG}'
for split in ['train','val','test']:
    n = len(list((RTTS / 'images' / split).iterdir()))
    print(f'RTTS    {split}: {n}')
for split in ['train','val','test']:
    n = len(list((VOCFOG / 'images' / split).iterdir()))
    print(f'VOC-FOG {split}: {n}')

# ── 3. Write YAML configs ─────────────────────────────────────────────────────
def write_yaml(path, dataset_root):
    with open(path, 'w') as f:
        f.write(f'path: {dataset_root}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('test: images/test\n')
        f.write('\nnames:\n')
        for i, cls in enumerate(['person','bicycle','car','bus','motorbike']):
            f.write(f'  {i}: {cls}\n')

write_yaml(WORK / 'rtts.yaml',   RTTS)
write_yaml(WORK / 'vocfog.yaml', VOCFOG)
print('rtts.yaml:')
print(open(WORK / 'rtts.yaml').read())

# ── 4. Sanity check modules ───────────────────────────────────────────────────
sys.path.insert(0, str(WORK))
import torch
from said.a2c2f_fsa import A2C2f_FSA
from said.wiou_loss import WIoU_v3_InnerMPDIoU

m   = A2C2f_FSA(64, 64, n_blocks=2, use_dsa=True)
out = m(torch.randn(2, 64, 80, 80))
assert out.shape == (2, 64, 80, 80)
print('A2C2f_FSA OK')

lf = WIoU_v3_InnerMPDIoU(0.6, 0.7); lf.train()
loss = lf(torch.tensor([[10.,10.,50.,50.]]), torch.tensor([[12.,12.,52.,52.]]))
assert loss.item() >= 0
print('WIoU_v3_InnerMPDIoU OK')

# ── 5. Verify SAID integration ────────────────────────────────────────────────
from said.integrate import setup_said, create_said_yaml
setup_said()
yaml_path = create_said_yaml(str(WORK / 'said_yolo11x.yaml'))
print(f'SAID YAML: {yaml_path}')

print('\nSetup complete. SAID modules integrated. Run training cells below.')
