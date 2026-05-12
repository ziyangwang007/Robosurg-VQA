# RoboSurg-VQA

RoboSurg-VQA is a multimodal benchmark for surgical segmentation-aware visual question answering (VQA). It converts surgical/endoscopic segmentation data into frame-level VQA annotations, where each image is paired with a fixed set of clinically motivated questions about surgical context, anatomy, imaging view, artefacts, visual quality, and visibility.

The current release focuses on an EndoVis 2017/2018 instantiation and provides scripts for model-assisted annotation using vision-language models, together with JSONL annotation outputs.

## Highlights

- Segmentation-aware surgical VQA benchmark for robotic/minimally invasive surgery.
- Fixed 9-question schema with controlled answer spaces.
- Uses RGB frames together with segmentation masks as spatial cues.
- Supports single-sample and batch annotation modes.
- Produces machine-readable JSON/JSONL outputs with answers and confidence scores.
- Includes dataset-specific annotations and a global taxonomy output format.

## Task definition

Given an image `I` and an associated segmentation mask `M`, the task is to answer a fixed set of 9 questions. Each answer is selected from a controlled label space, enabling consistent evaluation across models.

| ID | Question | Answer space |
| --- | --- | --- |
| Q1 | What is the primary surgical context in this frame? | `nephrectomy (robot-assisted)`, `renal/kidney-focused procedure`, `other abdominal procedure` |
| Q2 | Which organ or anatomical region does this image belong to? | `small intestine`, `kidney (parenchyma)`, `covered kidney (fat/fascia-covered kidney)`, `other abdominal soft tissue (peritoneum/mesentery/fat)` |
| Q3 | What is the imaging modality/view? | `endoscopic`, `robotic camera`, `synthetic`, `Unknown` |
| Q4 | Is there bleeding, smoke, or occlusion visible in the image? | 3-bit attribute: `bleeding`, `smoke`, `occlusion`, each in `{yes, no, Unknown}` |
| Q5 | How is the visual quality of the current scene? | `clear`, `blurry`, `reflective`, `mixed`, `Unknown` |
| Q6 | Is there specular reflection/glare? | `yes`, `no`, `Unknown` |
| Q7 | Is the image contrast normal? | `yes`, `no`, `Unknown` |
| Q8 | Is the centre of the frame occluded by instruments? | `yes`, `no`, `Unknown` |
| Q9 | Which regions contain smoke? | `none`, `left`, `right`, `centre`, `multi` |

## Dataset format

The annotation script expects the following directory structure:

```text
dataset_root/
├── site1/
│   ├── train/
│   │   ├── image/
│   │   │   ├── video1frame000.npy
│   │   │   └── ...
│   │   └── mask/
│   │       ├── video1frame000.npy
│   │       └── ...
│   └── test/
│       ├── image/
│       └── mask/
├── site2/
│   ├── train/
│   └── test/
└── site3/
    ├── train/
    └── test/
```

Images and masks should have one-to-one filename matching. For example:

```text
site1/train/image/video1frame000.npy
site1/train/mask/video1frame000.npy
```

The current EndoVis 2017/2018 instantiation contains 11,480 frames: 8,745 training frames and 2,735 test frames across three sites.

## Installation

Clone the repository:

```bash
git clone https://github.com/ziyangwang007/Robosurg-VQA.git
cd Robosurg-VQA
```

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install numpy pillow tqdm openai
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"  # macOS/Linux
# set OPENAI_API_KEY=your_api_key_here     # Windows CMD
# $env:OPENAI_API_KEY="your_api_key_here"  # Windows PowerShell
```

## Usage

### Dry run on a single sample

Use dry run to check whether the image and mask can be loaded correctly. This does not call the API.

```bash
python infer_endovis_single.py \
  --input path/to/image.npy \
  --gt path/to/mask.npy \
  --model gpt-4o-mini \
  --dry-run
```

### Annotate a single sample

```bash
python infer_endovis_single.py \
  --input path/to/image.npy \
  --gt path/to/mask.npy \
  --model gpt-4o-mini
```

The output will be saved to:

```text
outputs/answers.json
outputs/answers.global.json
```

### Batch annotation

```bash
python infer_endovis_single.py \
  --input-dir path/to/dataset_root \
  --out-dir outputs \
  --model gpt-4o-mini \
  --shard-size 10000 \
  --resume
```

Batch mode recursively scans:

```text
site*/train/image/*.npy
site*/train/mask/*.npy
site*/test/image/*.npy
site*/test/mask/*.npy
```

It then creates a manifest and writes sharded JSONL annotation files.

## Output files

Batch mode produces:

```text
outputs/
├── manifest.jsonl
├── state.json
├── annotations-00000.jsonl
├── annotations-00001.jsonl
├── global-annotations-00000.jsonl
└── global-annotations-00001.jsonl
```

- `annotations-*.jsonl`: dataset-specific annotation records.
- `global-annotations-*.jsonl`: records mapped to the global taxonomy.
- `manifest.jsonl`: generated sample manifest.
- `state.json`: checkpoint file for resume support.

A typical JSONL record has the following structure:

```json
{
  "id": "video1frame000",
  "site": "site1",
  "split": "train",
  "dataset": "EndoVis2017_2018",
  "task": "surgical_vqa_annotation",
  "model": "gpt-4o-mini",
  "image_id": "video1frame000.npy",
  "gt_id": "video1frame000.npy",
  "questions": {
    "q1": "What is the primary surgical context in this frame?",
    "q2": "Which organ or anatomical region does this image belong to?"
  },
  "answers": {
    "q1": {"value": "other abdominal procedure", "confidence": 0.8},
    "q2": {"value": "small intestine", "confidence": 0.9}
  },
  "error": null
}
```

The actual records contain all 9 questions and answers.

## Quality control

Before releasing annotations, we recommend running the following checks:

1. Every record has all 9 answers: `q1` to `q9`.
2. No answer falls outside the controlled label spaces.
3. If `q4.smoke == "no"`, then `q9 == "none"`.
4. The global sample key should be treated as `(site, split, id)` to avoid ambiguity.
5. Local absolute paths in `manifest.jsonl` and `state.json` should not be published as release metadata.
6. A random subset of frames should be manually audited for semantic plausibility and cross-question consistency.

## Important notes

- This repository does not redistribute the original EndoVis images or masks. Please obtain the source datasets from their official providers and follow their licence terms.
- The segmentation mask is used as a spatial cue, especially for visibility and centre-occlusion questions. It should not be used to infer non-visible case-specific facts.
- The current script is designed for `.npy` image and mask files. If your dataset uses PNG/JPEG images or colour-coded masks, add a conversion step before running the annotation script.
- For public release, avoid committing API keys, `.env` files, raw dataset files, cache folders, or local machine paths.

## Recommended repository structure

```text
Robosurg-VQA/
├── README.md
├── infer_endovis_single.py
├── requirements.txt
├── outputs/
│   ├── annotations-00000.jsonl
│   ├── annotations-00001.jsonl
│   ├── global-annotations-00000.jsonl
│   └── global-annotations-00001.jsonl
├── docs/
│   └── figures/
├── scripts/
│   └── validate_annotations.py
├── .gitignore
└── LICENSE
```

Suggested `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
.env
.venv/
raw_data/
data/
*.npy
outputs/state.json
outputs/manifest.jsonl
```

Depending on the release plan, you may choose to track only the final annotation JSONL files and keep raw images/masks outside the repository.

## Citation

If you use RoboSurg-VQA, please cite the accompanying paper:

```bibtex
@inproceedings{robosurgvqa2026,
  title     = {RoboSurg-VQA: A Multimodal Benchmark for Surgical Segmentation-Aware Visual Question Answering},
  author    = {To be updated},
  booktitle = {To be updated},
  year      = {2026}
}
```

## Acknowledgements

RoboSurg-VQA builds on public surgical segmentation resources, including EndoVis 2017 and EndoVis 2018. We thank the original dataset creators and challenge organisers for making these resources available to the community.

## Contact

For questions or suggestions, please open an issue in this repository.
