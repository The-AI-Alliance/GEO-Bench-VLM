<h1>This repository provides the <strong>PyTorch implementation</strong> of our work <strong>GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks</strong>.</h1>


### Environment Setup

🔗 Access and set up the following vision-language models: [Qwen2.5-VL (QwenLM)](https://github.com/QwenLM/Qwen2.5-VL), [InternVL2 (OpenGVLab)](https://github.com/OpenGVLab/InternVL), [LLaVA 1.5 / 1.6 (haotian-liu)](https://github.com/haotian-liu/LLaVA), and [LLaVA-OneVision (LLaVA-VL)](https://github.com/LLaVA-VL/LLaVA-NeXT). Each repository includes environment setup instructions.

<hr>

### 🔻 Model Weights Download

Download the following pretrained model weights and place them in the `Out_weights/` folder:

- [LLaVA 1.6 (Vicuna-7B)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)  
- [LLaVA 1.5 (Vicuna-7B)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)  
- [InternVL2 (8B)](https://huggingface.co/OpenGVLab/InternVL2-8B)  
- [Qwen2-VL (7B Instruct)](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)  
- [LLaVA-OneVision (llava-onevision)](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-si)

📁 After downloading, ensure all models are stored under the `Out_weights/` directory for proper loading during inference.

<hr>

All downloaded folders should follow this structure:

```
Out_weights/
├── llava-1.5-7b-hf/
├── llava-v1.6-vicuna-7b-hf/
├── InternVL2-8B/
├── Qwen2-VL-7B-Instruct/
├── llava-onevision-qwen2-7b-si/
```

### For Single Image

Now follow the command given from folder <strong>[eval_geobenchvlm](eval_geobenchvlm)</strong>

```
python runmodel.py <model-key> --data_path /datasets/GEOBench-VLM
```
Available <model-key> options:

| Model Name      | Model Key   |
| --------------- | ----------- |
| LLaVA 1.5       | `llava1pt5` |
| LLaVA 1.6       | `llava1pt6` |
| LLaVA-OneVision | `llavaone1` |
| Qwen2-VL        | `qwen`      |
| InternVL2       | `internvl`  |

Example:

```
python runmodel.py qwen --data_path /datasets/GEOBench-VLM
```

### For Temporal

Now follow the command given from folder <strong>[eval_geobenchvlm temporal](eval_geobenchvlm/temporal)</strong>

Available <model-key> options:

| Model Name      | Model Key   |
| --------------- | ----------- |
| Qwen2-VL        | `qwen`      |
| LLaVA-OneVision | `llavaone1` |

```
python run_temporal.py qwen --data_path /datasets/GEOBench-VLM
```



