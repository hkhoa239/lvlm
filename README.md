# Where do Large Vision-Language Models Look at when Answering Questions?
The official repo for "[Where do Large Vision-Language Models Look at when Answering Questions?](https://arxiv.org/pdf/2503.13891)" A PyTorch implementation for a salieny heatmap visualization method that interprets the open-ended responses of LVLMs conditioned on an image.

### Installation
First clone this repository and navigate to the folder.

The environment installation mainly follows [LLaVA](https://github.com/haotian-liu/LLaVA). You can update the pip and install the dependencies using:

```
$ pip install --upgrade pip
$ bash install.sh
```

### Model Preparation
For Mini-Gemini models, please follow the instructions in [MGM](https://github.com/dvlab-research/MGM) to download the models and put them in the folders following [Structure](https://github.com/dvlab-research/MGM?tab=readme-ov-file#structure)

### Quick Start
To generate the saliency heatmap of an LVLM when generating free-form responses, an example command is as follows, with the hyperparameters passed as arguments:
```
$ python3 main.py --method iGOS+ --model llava --dataset <dataset name> --data_path <path/to/questions> --image_folder <path/to/images> --output_dir <path/to/output> --size 32 --L1 1.0 --L2 0.1 --L3 10.0 --ig_iter 10 --gamma 1.0 --iterations 5 --momentum 5
```
The explanations of each argument can be found in [args.py](args.py)

### Datasets
You may find the datasets at [https://huggingface.co/datasets/xiaoying0505/LVLM_Interpretation](https://huggingface.co/datasets/xiaoying0505/LVLM_Interpretation) to reproduce the results in the paper.

### Acknowledgement
Some parts of the code are built upon [IGOS_pp](https://github.com/khorrams/IGOS_pp). And we use the open-source LVLMs [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [Cambrian](https://github.com/cambrian-mllm/cambrian) and [Mini-Gemini](https://github.com/dvlab-research/MGM) in this project. We thank the authors for their excellent work.
