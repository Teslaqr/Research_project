"""
Generates saa_bottle_demo.ipynb.

Edit cells here, then regenerate with:
    python kaggle/generate_notebook.py
Do not hand-edit the .ipynb JSON directly.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# SAA+ on MVTec-AD — bottle category demo

Runs [Segment Any Anomaly (SAA+)](https://github.com/caoyunkang/Segment-Any-Anomaly)
(training-free anomaly segmentation via Grounding DINO + SAM) on the `bottle`
category of MVTec-AD.

**Before running:** Notebook settings (right sidebar) →
- Accelerator: **GPU T4 x2** (or any available GPU)
- Internet: **On**
- Add Input → dataset **`ipythonx/mvtec-ad`** (or another MVTec-AD mirror /
  your own private upload — anything containing a `bottle/` folder with
  `train/`, `test/`, `ground_truth/` subfolders)
"""))

cells.append(nbf.v4.new_code_cell("!nvidia-smi"))

cells.append(nbf.v4.new_markdown_cell("## 1. Clone SAA+"))
cells.append(nbf.v4.new_code_cell(
"""import os, subprocess

PROJECT_DIR = '/kaggle/working/Segment-Any-Anomaly'

if not os.path.isdir(PROJECT_DIR):
    subprocess.run(
        ['git', 'clone', '--depth', '1',
         'https://github.com/caoyunkang/Segment-Any-Anomaly.git', PROJECT_DIR],
        check=True,
    )
else:
    print('Repo already cloned, skipping.')

os.chdir(PROJECT_DIR)
print('cwd ->', os.getcwd())
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Install dependencies

We deliberately skip `install.sh`'s pinned `torch==1.10+cu113` — Kaggle
already ships a recent torch/CUDA pair, and Grounding DINO's CUDA extension
needs to be built against whatever torch is actually installed, not an old
pin. `requirements.txt` inside `GroundingDINO/` doesn't pin a torch version,
so this is safe."""))
cells.append(nbf.v4.new_code_cell(
"""import torch
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())

%cd /kaggle/working/Segment-Any-Anomaly
!pip install -q -e ./GroundingDINO
!pip install -q -e ./SAM
!pip install -q opencv-python pycocotools matplotlib onnxruntime onnx \\
    transformers addict yapf timm loguru tqdm scikit-image scikit-learn \\
    pandas seaborn open_clip_torch einops "supervision==0.3.2"
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Download model weights (SAM ViT-H, Grounding DINO SwinT)"))
cells.append(nbf.v4.new_code_cell(
"""import os

os.makedirs('weights', exist_ok=True)
sam_ckpt = 'weights/sam_vit_h_4b8939.pth'
dino_ckpt = 'weights/groundingdino_swint_ogc.pth'

if not os.path.exists(sam_ckpt):
    !wget -q -O {sam_ckpt} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
if not os.path.exists(dino_ckpt):
    !wget -q -O {dino_ckpt} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

print('SAM weights:  %.1f MB' % (os.path.getsize(sam_ckpt) / 1e6))
print('DINO weights: %.1f MB' % (os.path.getsize(dino_ckpt) / 1e6))
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. Link the bottle-only dataset

SAA+ expects data at `../datasets/mvtec_anomaly_detection/<category>/` relative
to the project root. We auto-detect wherever the attached Kaggle dataset put
a `bottle/` folder (with `train/` + `test/` subfolders) and symlink it into
place — this works regardless of which MVTec-AD mirror dataset you attach."""))
cells.append(nbf.v4.new_code_cell(
"""import os, glob

candidates = glob.glob('/kaggle/input/**/bottle', recursive=True)
bottle_src = next(
    (c for c in candidates
     if os.path.isdir(os.path.join(c, 'train')) and os.path.isdir(os.path.join(c, 'test'))),
    None,
)

if bottle_src is None:
    raise FileNotFoundError(
        "No 'bottle' folder with train/test subfolders found under /kaggle/input. "
        "Attach a dataset containing MVTec-AD's bottle category via 'Add Input' "
        "(e.g. ipythonx/mvtec-ad), then re-run this cell."
    )

print('Found bottle data at:', bottle_src)

DATASET_ROOT = '/kaggle/working/datasets/mvtec_anomaly_detection'
os.makedirs(DATASET_ROOT, exist_ok=True)
dst = os.path.join(DATASET_ROOT, 'bottle')
if not os.path.exists(dst):
    os.symlink(bottle_src, dst)

assert os.path.isdir(os.path.join(dst, 'ground_truth')), (
    "This dataset's bottle folder has no ground_truth/ subfolder, which SAA+ needs "
    "for pixel-level evaluation. Fallback: download the official archive from "
    "https://www.mvtec.com/research-teaching/datasets/mvtec-ad, extract just the "
    "'bottle' folder, upload it as a new private Kaggle dataset, attach it here, "
    "and re-run this cell."
)
print('Linked to:', dst, '->', os.listdir(dst))
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 5. Sanity check — one image

Runs the model on a single defective bottle image first. If this cell fails,
fix it before burning time on the full test-set evaluation below."""))
cells.append(nbf.v4.new_code_cell(
"""import sys, cv2, glob
import matplotlib.pyplot as plt

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
import SAA as SegmentAnyAnomaly

device = 'cuda:0'
model = SegmentAnyAnomaly.Model(
    dino_config_file='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    dino_checkpoint='weights/groundingdino_swint_ogc.pth',
    sam_checkpoint='weights/sam_vit_h_4b8939.pth',
    box_threshold=0.1,
    text_threshold=0.1,
    out_size=1024,
    device=device,
)

textual_prompts = ['broken part. contamination. white broken.', 'bottle']
property_text_prompts = (
    'the image of bottle have 1 dissimilar bottle, with a maximum of 5 anomaly. '
    'The anomaly would not exceed 0.3 object area. '
)
model.set_ensemble_text_prompts(textual_prompts, verbose=False)
model.set_property_text_prompts(property_text_prompts, verbose=False)
model = model.to(device)

test_images = glob.glob('../datasets/mvtec_anomaly_detection/bottle/test/broken_large/*.png')
assert test_images, 'No test images found — check the dataset-linking cell above.'
image = cv2.imread(test_images[0], cv2.IMREAD_COLOR)
score, appendix = model(image)
similarity_map = appendix['similarity_map']

image_show = cv2.cvtColor(cv2.resize(image, (1024, 1024)), cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_show); axes[0].set_title('Input'); axes[0].axis('off')
axes[1].imshow(image_show); axes[1].imshow(cv2.resize(score, (1024, 1024)), alpha=0.4, cmap='jet')
axes[1].set_title('Anomaly Score'); axes[1].axis('off')
axes[2].imshow(image_show); axes[2].imshow(cv2.resize(similarity_map, (1024, 1024)), alpha=0.4, cmap='jet')
axes[2].set_title('Saliency'); axes[2].axis('off')
plt.tight_layout()
plt.savefig('/kaggle/working/sanity_check.png', dpi=120)
plt.show()
print('Sanity check passed.')
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 6. Full evaluation on the bottle test set

Runs the repo's own `eval_SAA.py` (zero-shot, no training) over every bottle
test image and computes image/pixel-level AUROC. Run as a subprocess so we
capture full stdout/stderr for debugging if it errors."""))
cells.append(nbf.v4.new_code_cell(
"""import subprocess

os.chdir(PROJECT_DIR)
cmd = [
    'python', 'eval_SAA.py',
    '--dataset', 'mvtec',
    '--class-name', 'bottle',
    '--batch-size', '1',
    '--root-dir', '/kaggle/working/result',
    '--cal-pro', 'False',
    '--gpu-id', '0',
]
print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print('--- STDOUT (tail) ---')
print('\\n'.join(result.stdout.splitlines()[-60:]))
print('--- STDERR (tail) ---')
print('\\n'.join(result.stderr.splitlines()[-60:]))
print('Return code:', result.returncode)
assert result.returncode == 0, 'eval_SAA.py failed — see STDERR above.'
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Results"))
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd, glob

csv_files = glob.glob('/kaggle/working/result/**/*.csv', recursive=True)
print('CSV files found:', csv_files)
for f in csv_files:
    print(f)
    display(pd.read_csv(f, index_col=0))
"""))
cells.append(nbf.v4.new_code_cell(
"""import matplotlib.image as mpimg

vis_images = sorted(glob.glob('/kaggle/working/result/**/*.png', recursive=True))
sample = vis_images[:6]
if sample:
    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    axes = [axes] if len(sample) == 1 else axes
    for ax, path in zip(axes, sample):
        ax.imshow(mpimg.imread(path)); ax.axis('off'); ax.set_title(os.path.basename(path), fontsize=8)
    plt.tight_layout()
    plt.show()
else:
    print('No visualization images found under /kaggle/working/result.')
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Next steps

- To try another category, repeat cell 4 for e.g. `hazelnut`/`screw`, and change
  `--class-name` in cell 6.
- To run the whole MVTec-AD suite, attach the full dataset and use the repo's
  own `run_MVTec.py` instead of the single-class command above.
"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}

with open('kaggle/saa_bottle_demo.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Wrote kaggle/saa_bottle_demo.ipynb')
