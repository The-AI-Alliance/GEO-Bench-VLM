# GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Muhammad Sohail Danish](https://www.linkedin.com/in/muhammad-sohail-danish/)\*, [Muhammad Akhtar Munir](https://akhtarvision.github.io/)\*, [Syed Roshaan Ali Shah](https://www.linkedin.com/in/syed-roshaan-ali-shah-b797b44a/), [Kartik Kuckreja](https://www.linkedin.com/in/kartik-kuckreja-930531221/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Paolo Fraccaro](https://research.ibm.com/people/paolo-fraccaro) , [Alexandre Lacoste](https://www.servicenow.com/research/author/alexandre-lacoste.html)   and [Salman Khan](https://salman-h-khan.github.io/)
\* Equally contributing first authors

#### **Mohamed bin Zayed University of AI, University College London, Linköping University, IBM Research Europe, UK, ServiceNow Research, Australian National University**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.19325)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](http://the-ai-alliance.github.io/GEO-Bench-VLM)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/aialliance/GEOBench-VLM)


Official GitHub repository for  `GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks`.
---

## 📢 Latest Updates
- **Dec-02-24**: We release the benchmark dataset [huggingface link](https://huggingface.co/datasets/aialliance/GEOBench-VLM). 
- **Dec-02-24**: Arxiv Preprint is released [arxiv link](https://arxiv.org/abs/2411.19325). 🔥🔥
---

## 🛠️ Code and Leaderboard Coming Soon!

The code and leaderboard will be released shortly. Follow this repository for updates!

---

## 💡 Overview
<p align="center">
   <img src="images/teaser_bench.jpg" width="1200"></a>
</p>
<p align="justify"> <b> Figure</b>: Examples of tasks from the GEOBench-VLM benchmark. Our benchmark is designed to evaluate VLMs on a diverse range of remote sensing applications. The benchmark includes over 10,000 questions spanning a range of tasks essential for Earth Observation, such as Temporal Understanding, Referring Segmentation, Visual Grounding, Scene Understanding, Counting, Detailed Image Captioning, and Relational Reasoning. Each task is tailored to capture unique domain-specific challenges, featuring varied visual conditions and object scales, and requiring nuanced understanding for applications like disaster assessment, urban planning, and environmental monitoring.

**<p align="justify"> Abstract:** While numerous recent benchmarks focus on evaluating generic Vision-Language Models (VLMs), they fall short in addressing the unique demands of geospatial applications. 
Generic VLM benchmarks are not designed to handle the complexities of geospatial data, which is critical for applications such as environmental monitoring, urban planning, and disaster management.
Some of the unique challenges in geospatial domain include temporal analysis for changes, counting objects in large quantities, detecting tiny objects, and understanding relationships between entities occurring in Remote Sensing imagery. 
To address this gap in the geospatial domain, we present GEOBench-VLM, a comprehensive benchmark specifically designed to evaluate VLMs on geospatial tasks, including scene understanding, object counting, localization, fine-grained categorization, and temporal analysis. 
Our benchmark features over 10,000 manually verified instructions and covers a diverse set of variations in visual conditions, object type, and scale.
We evaluate several state-of-the-art VLMs to assess their accuracy within the geospatial context. 
The results indicate that although existing VLMs demonstrate potential, they face challenges when dealing with geospatial-specific examples, highlighting the room for further improvements. 
Specifically, the best-performing GPT4o achieves only 40\% accuracy on MCQs, which is only double the random guess performance. </p>

## 🏆 Contributions

- **GEOBench-VLM Benchmark.** We introduce GEOBench-VLM, a benchmark suite designed specifically for evaluating VLMs on geospatial tasks, addressing geospatial data challenges. It covers 8 broad categories and 31 sub-tasks with over 10,000 manually verified questions.
- **Evaluation of VLMs.**We provide a detailed evaluation of ten state-of-the-art VLMs, including generic (open and closed-source) and task-specific geospatial VLMs, highlighting their capabilities and limitations in handling geospatial tasks.
- **Analysis of Geospatial Task Performance.** We analyze performance across a range of tasks, including scene classification, counting, change detection, relationship prediction, visual grounding, image captioning, segmentation, disaster detection, and temporal analysis, among others, providing key insights into improving VLMs for geospatial applications.
<hr />

## 🗂️ Benchmarks Comparison

<p align="center">
   <img src="images/benchmark-table.png" width="1200" alt="Dataset Comparison table"></a>
</p>


> <p align="justify"> <b> <span style="color: blue;">Table</span></b>: Overview of Generic and Geospatial-specific Datasets & Benchmarks, detailing modalities (O=Optical, PAN=Panchromatic, MS=Multi-spectral, IR=Infrared, SAR=Synthetic Aperture Radar, V=Video, MI=Multi-image, BT=Bi-Temporal, MT=Multi-temporal), data sources (DRSD=Diverse RS Datasets, OSM=OpenStreetMap, GE=Google Earth, answer types (MCQ=Multiple Choice, SC=Single Choice, FF=Free-Form, BBox=Bounding Box, Seg=Segmentation Mask), and annotation types (A=Automatic, M=Manual). </p>

<hr />

## 🔍 Dataset Annotation Pipeline
<p align="justify"> Our pipeline integrates diverse datasets, automated tools, and manual annotation. Tasks such as scene understanding, object classification, and non-optical analysis are based on classification datasets, while GPT-4o generates unique MCQs with five options: one correct answer, one semantically similar "closest" option, and three plausible alternatives. Spatial relationship tasks rely on manually annotated object pair relationships, ensuring consistency through cross-verification. Caption generation leverages GPT-4o, combining image, object details, and spatial interactions with manual refinement for high precision. </p>
<p align="center">
   <img src="images/pipeline7.jpg" width="1200"></a>
</p>



<hr />

## 📊 Results

Performance Summary of VLMs Across Geospatial Tasks. GPT-4o achieves better accuracy in relatively easy tasks like Aircraft
Type Classification, Disaster Type Classification, Scene Classification, and Land Use Classification. But, on average the best-performing
GPT-4o achieves only **40%** accuracy on MCQs based on diverse geospatial tasks, which is only double the random guess performance.
These results showcase the varying strengths of VLMs in addressing diverse geospatial tasks.
<p align="center">
   <img src="images/benchmark_heatmap1.png" width="1200" alt="Results Heatmap"></a>
</p>


### Temporal Understanding Results

Results highlight the strengths of VLMs in handling temporal geospatial challenges. Evaluation across five tasks: Crop Type Classification, Disaster Type Classification, Farm Pond Change Detection, Land Use Classification, and Damaged Building Count. GPT-4o achieves the highest accuracy overall in classification and counting tasks.

<div align="center">

| **Model**            | **Crop Type Classification** | **Disaster Type Classification** | **Farm Pond Change Detection** | **Land Use Classification** | **Damaged Building Count** |
|---------------------|:---------------------------:|:--------------------------------:|:------------------------------:|:---------------------------:|:---------------------------:|
| LLaVA-OneV           | 0.1273                      | 0.4493                          | **0.1579**                     | 0.5672                      | 0.2139                      |
| Qwen2-VL             | 0.1273                      | 0.5903                          | 0.0921                         | 0.5869                      | 0.2270                      |
| GPT-4o               | **0.1818**                  | **0.6344**                      | 0.1447                         | **0.6230**                  | **0.2420**                  |

</div>

### Reffering Expression Detection
Referring expression detection. We report Precision on 0.5 IoU and 0.25 IoU
<div align="center">

| **Model**            | **Precision@0.5 IoU** | **Precision@0.25 IoU** |
|:---------------------:|:--------------------:|:----------------------:|
| Sphinx               | 0.3408               | 0.5289                 |
| GeoChat              | 0.1151               | 0.2100                 |
| Ferret               | 0.0943               | 0.2003                 |
| Qwen2-VL             | 0.1518               | 0.2524                 |
| GPT-4o               | 0.0087               | 0.0386                 |

</div>



<hr />

## 🤖 Qualitative Results

> <p align="justify"> <b> <span style="color: blue;">Scene Understanding</span></b>: This illustrates model performance on geospatial scene understanding tasks, highlighting successes in clear contexts and challenges in ambiguous scenes. The results emphasize the importance of contextual reasoning and addressing overlapping visual cues for accurate classification.
<p align="center">
  <img src="images/results-scene.jpg"  width="1200" alt="Scene Understanding">
</p>

> <p align="justify"> <b> <span style="color: blue;">Counting</span></b>: The figure showcases model performance on counting tasks, where Qwen 2-VL, GPT-4o and LLaVA-One have better performance in identifying objects. Other models, such as Ferret, struggled with overestimation, highlighting challenges in object differentiation and spatial reasoning.

<p align="center">
  <img src="images/results-counting.jpg"  width="1200" alt="Counting">
</p>

> <p align="justify"> <b> <span style="color: blue;">Object Classification</span></b>: The figure highlights model performance on object classification, showing success with familiar objects like the "atago-class destroyer" and "small civil transport/utility" aircraft. However, models struggled with rarer objects like the ``murasame-class destroyer" and ``garibaldi aircraft carrier" indicating a need for improvement on less common classes and fine-grained recognition.

<p align="center">
  <img src="images/results-object.jpg"  width="1200" alt="Object Classification">
</p>

> <p align="justify"> <b> <span style="color: blue;">Event Detection</span></b>: Model performance on disaster assessment tasks, with success in scenarios like 'fire' and 'flooding' but challenges in ambiguous cases like 'tsunami' and 'seismic activity'. Misclassifications highlight limitations in contextual reasoning and insufficient exposure on overlapping disaster features.

<p align="center">
  <img src="images/results-event.jpg"  width="1200" alt="Event Detection">
</p>

> <p align="justify"> <b> <span style="color: blue;">Spatial Relations</span></b>: The figure demonstrates model performance on spatial relationship tasks, with success in close-object scenarios and struggles in cluttered environments with distant objects.

<p align="center">
  <img src="images/results-relations.jpg"  width="1200" alt="Spatial Relations">
</p>



<hr />

## 📜 Citation
If you find our work and this repository useful, please consider giving our repo a star and citing our paper as follows:
```bibtex
@article{danish2024geobenchvlm,
      title={GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks}, 
      author={Muhammad Sohail Danish and Muhammad Akhtar Munir and Syed Roshaan Ali Shah and Kartik Kuckreja and Fahad Shahbaz Khan and Paolo Fraccaro and Alexandre Lacoste and Salman Khan},
      year={2024},
      eprint={2411.19325},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.19325}, 
}
```




## 📨 Contact
If you have any questions, please create an issue on this repository or contact at muhammad.sohail@mbzuai.ac.ae.

---
[<img src="images/MBZUAI_logo.png" width="290" height="85">](https://mbzuai.ac.ae)
[<img src="images/IVAL_logo.png" width="160" height="100">](https://www.ival-mbzuai.com)
[<img src="images/ibm-logo.jpg" width="270">](https://research.ibm.com/labs/uk)
[<img src="images/ServiceNow_logo.png" width="270">](https://www.servicenow.com/)
[<img src="images/aialliance.png" width="270">](https://thealliance.ai/)
