This repository contains a collection of resources and papers on ***Vision-Language Models***.

## Contents
- [Papers](#papers)
  - [Survey](#survey)
  - [Vision](#vision)
    - [Object Detection](#object-detection)
    - [Classification](#classification)
    - [Generation \& Manipulation](#generation--manipulation)
    - [Segmentation](#segmentation)
    - [3D](#3d)
    - [Video](#video)
    - [Face](#face)
  - [Miscellaneous](#miscellaneous)
  - [Natural Language](#natural-language)
  - [Theory](#theory)
  - [Medical Image](#medical-image)



# Papers

## Survey

**Vision-Language Pre-training: Basics, Recent Advances, and Future Trends**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2210.09263)]

## Vision
### Object Detection

**Open-vocabulary Object Detection via Vision and Language Knowledge Distillation** \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.13921)][[Github](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)]

**RegionCLIP: Region-based Language-Image Pretraining** \
CVPR 2022.
[[Paper](https://arxiv.org/abs/2112.09106)][[Github](https://github.com/microsoft/RegionCLIP)]


**Grounded Language-Image Pre-training** \
CVPR 2022.[[Paper](https://arxiv.org/abs/2112.03857)][[Gitub](https://github.com/microsoft/GLIP)]

**Detecting Twenty-thousand Classes using Image-level Supervision** \
ECCV 2022.[[Paper](https://arxiv.org/abs/2201.02605)][[Github](https://github.com/facebookresearch/Detic)]


**PromptDet: Towards Open-vocabulary Detection using Uncurated Images** \
ECCV 2022.[[Paper](https://arxiv.org/abs/2203.16513)][[Github](https://github.com/fcjian/PromptDet)]


**Simple Open-Vocabulary Object Detection with Vision Transformers** \
ECCV 2022 [[Paper](https://arxiv.org/abs/2205.06230)][[Github](https://github.com/google-research/scenic)]

**Open-Vocabulary DETR with Conditional Matching** \
ECCV 2022.[[Paper](https://arxiv.org/abs/2203.11876)][[Github](https://github.com/yuhangzang/OV-DETR)]

**X-DETR: A Versatile Architecture for Instance-wise Vision-Language Tasks** \
ECCV 2022.[[Paper](https://arxiv.org/abs/2204.05626)]


**Pix2seq: A Language Modeling Framework for Object Detection** \
ICLR 2022.[[Paper](https://arxiv.org/abs/2109.10852)][[Github](https://github.com/google-research/pix2seq)]

**F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2209.15639)]

**Learning Object-Language Alignments for Open-Vocabulary Object Detection**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2211.14843)][[Github](https://github.com/clin1223/VLDet)]

**ProposalCLIP: Unsupervised Open-Category Object Proposal Generation via Exploiting CLIP Cues**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2201.06696)]

**CLIP the Gap: A Single Domain Generalization Approach for Object Detection**\
arXiv 2023.[[Paper](https://arxiv.org/pdf/2301.05499.pdf)]

### Classification

**K-LITE: Learning Transferable Visual Models with External Knowledge**\
NeurlPS 2022.[[Paper](https://arxiv.org/abs/2204.09222)]

**Visual Classification via Description From Large Language Models** \
ICLR 2023.[[Paper](https://openreview.net/pdf?id=jlAjNL8z5cs)]

**Learning to Compose Soft Prompts for Compositional Zero-Shot Learning**\
ICLR 2023. [[Paper](https://arxiv.org/abs/2204.03574)][[Github](https://github.com/BatsResearch/csp)]

**Masked Unsupervised Self-training for Zero-shot Image Classification**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2206.02967)][[Github](https://github.com/salesforce/MUST)]

**CLIPood: Generalizing CLIP to Out-of-Distributions**\
arXiv 2023.[[Paper](https://arxiv.org/abs/2302.00864)]


**ON-THE-FLY OBJECT DETECTION USING STYLEGAN WITH CLIP GUIDANCE**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2210.16742)]

### Generation & Manipulation
**ISS: Image as Stepping Stone for Text-Guided 3D Shape Generation** \
ICLR 2023.[[Paper](https://arxiv.org/abs/2209.04145)]

**An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion** \
ICLR 2023. [[Paper](https://arxiv.org/abs/2208.01618)][[Github](https://github.com/rinongal/textual_inversion)]

**CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers** \
ICLR 2023.[[Paper](https://arxiv.org/abs/2205.15868)][[Github](https://github.com/THUDM/CogVideo)]

**Learning Input-Agnostic Manipulation Directions in StyleGAN with Text Guidance**\
ICLR 2023.[[Paper](https://openreview.net/pdf?id=47B_ctC4pJ)]

**CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2110.02624)][[Github](https://github.com/AutodeskAILab/Clip-Forge)]

**MotionCLIP: Exposing Human Motion Generation to CLIP Space**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2203.08063#:~:text=MotionCLIP%20comprises%20a%20transformer%2Dbased,in%20a%20self%2Dsupervised%20manner.)][[Github](https://github.com/GuyTevet/MotionCLIP)]

**VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2204.08583)][[Github](https://github.com/EleutherAI/vqgan-clip)]

**CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders**\
NeurIPS 2022.[[Paper](https://arxiv.org/abs/2106.14843)]

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**\
arXiv 2023.[[Paper](https://arxiv.org/abs/2301.12959)]


### Segmentation
**CRIS: CLIP-Driven Referring Image Segmentation**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2111.15174)][[Github](https://github.com/DerrickWang005/CRIS.pytorch)]

**Extract Free Dense Labels from CLIP**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2112.01071)][[Github](https://github.com/chongzhou96/MaskCLIP)]

**Open-world Semantic Segmentation via Contrasting and Clustering Vision-Language Embedding**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2207.08455)]

**A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-language Model**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2112.14757)]

**CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.09506v2)]

**Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2210.04150)]

**Image Segmentation Using Text and Image Prompts**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2112.10003)][[Github](https://github.com/timojl/clipseg)]

**MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2208.12262)]

**DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2112.01518)]

**GroupViT: Semantic Segmentation Emerges from Text Supervision**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2202.11094)]

### 3D
**PointCLIP: Point Cloud Understanding by CLIP**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2112.02413)][[Github](https://github.com/ZrrSkywalker/PointCLIP)]

**CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP**\
arXiv 2023.[[Paper](https://arxiv.org/abs/2301.04926)]

**PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2211.11682)]

**CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2210.05663)]

### Video
**CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2209.06430)]

**Frozen CLIP Models are Efficient Video Learners**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2208.03550)][[Github](https://github.com/OpenGVLab/efficient-video-recognition)]

**Zero-Shot Temporal Action Detection via Vision-Language Prompting**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2207.08184)]

**FitCLIP: Refining Large-Scale Pretrained Image-Text Models for Zero-Shot Video Understanding Tasks**\
BMVC 2022.[[Paper](https://arxiv.org/abs/2203.13371)]

**Fine-tuned CLIP Models are Efficient Video Learners**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.03640)]

### Face
**Face Recognition in the age of CLIP & Billion image datasets**\
arXiv 2023.[[Paper](https://arxiv.org/abs/2301.07315)]

## Miscellaneous

**Conditional Prompt Learning for Vision-Language Models**\
CVPR 2022.[[Paper](https://arxiv.org/abs/2203.05557)][[Github](https://github.com/KaiyangZhou/CoOp)]

**Prompt Learning with Optimal Transport for Vision-Language Models**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2210.01253)]

**Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2206.08916)][[Github](https://github.com/allenai/unified-io-inference)]

**"This is my unicorn, Fluffy":Personalizing frozen vision-language representations**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2204.01694)]


**Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone**\
NeurIPS 2022.[[Paper](https://arxiv.org/abs/2206.07643)]

**BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**\
NeurIPS 2022.[[Paper](https://arxiv.org/abs/2201.12086)]


**Attentive Mask CLIP**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.09506v2)]

**CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.06138)]

**Frozen CLIP Model is An Efficient Point Cloud Backbone**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.04098)]

## Natural Language
**Linearly Mapping from Image to Text Space**\
ICLR 2023.[[Paper](https://arxiv.org/abs/2209.15162)]

**DECAP: Decoding CLIP Latents for Zero-shot Captioning**\
ICLR 2023.[[Paper](https://openreview.net/forum?id=Lt8bMlhiwx2)]

**Weakly Supervised Grounding for VQA in Vision-Language Transformers**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2207.02334)]

**UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2111.12085)]


## Theory
**When and why vision-language models behave like bags-of-words, and what to do about it?** \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.01936)][[Code Coming Soon](https://github.com/mertyg/when-and-why-vlms-bow)]

**Generative Negative Text Replay for Continual Vision-Language Pretraining**\
ECCV 2022.[[Paper](https://arxiv.org/abs/2210.17322)]

**Does CLIP Bind Concepts? Probing Compositionality in Large Image Models**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.10537)]

**When are Lemons Purple? The Concept Association Bias of CLIP**\
arXiv 2022.[[Paper](https://arxiv.org/abs/2212.12043)]

## Medical Image
**CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection**\
arXiv 2023.[[Paper](https://arxiv.org/abs/2301.00785)]
