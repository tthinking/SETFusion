
#  SETFusion: A Semantic Transformer for Infrared and Visible Image Fusion (PR 2026)

This is the official implementation of the SETFusion model proposed in the paper ([SETFusion: A Semantic Transformer for Infrared and Visible Image Fusion ]([https://www.sciencedirect.com/science/article/pii/S0031320324005739?via%3Dihub](https://authors.elsevier.com/sd/article/S0031-3203(26)00093-2))) with Pytorch.


### Comparison on the downstream task, i.e., object detection.
![Image](figs/objectDetection.png)


### Visualizations of the feature maps from the proposed PSTM and MSTM modules.
![Image](figs/featureMaps.png)



### Comparison with SOTA image fusion modules.
| Methods     | End-to-End | Convolutional Operation | Pyramid Semantic Transformer | Multi-scale Semantic Transformer | VIF Loss | Unsupervised | Generalization Ability |
|:-----------:|:----------:|:------------------------:|:-----------------------------:|:--------------------------------:|:--------:|:------------:|:-----------------------:|
| GTF         | ✘          | ✘                        | ✘                             | ✘                                | ✘        | ✘            | ✘                       |
| RFN-Nest    | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✘            | ✔                       |
| PMGI        | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✘                       |
| FusionGAN   | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✘                       |
| MFEIF       | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✔                       |
| GANMcC      | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✔                       |
| PPT Fusion  | ✘          | ✘                        | ✘                             | ✘                                | ✘        | ✘            | ✘                       |
| DATFuse     | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✔                       |
| TCCFusion   | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✔                       |
| CrossFuse   | ✘          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✘                       |
| MMDRFuse    | ✔          | ✔                        | ✘                             | ✘                                | ✘        | ✔            | ✔                       |
| SETFusion   | ✔          | ✔                        | ✔                             |  ✔                               |  ✔        | ✔            | ✔                       |


### A detailed quantitative comparison of different methods on TNO dataset

<p align="center">
  <img src="figs/TNO/Q_SF.jpg" width="49%">
  <img src="figs/TNO/LMI.jpg" width="49%">
</p>
<p align="center">
  <img src="figs/TNO/AG.jpg" width="49%">
  <img src="figs/TNO/EI.jpg" width="49%">
</p>
<p align="center">
  <img src="figs/TNO/N_ab_f.jpg" width="49%">
  <img src="figs/TNO/DF.jpg" width="49%">
</p>


### A detailed quantitative comparison of different methods on RoadScene dataset

<p align="center">
  <img src="figs/Q_SF.jpg" width="49%">
  <img src="figs/LMI.jpg" width="49%">
</p>
<p align="center">
  <img src="figs/AG.jpg" width="49%">
  <img src="figs/EI.jpg" width="49%">
</p>
<p align="center">
  <img src="figs/N_ab_f.jpg" width="49%">
  <img src="figs/DF.jpg" width="49%">
</p>



### Computational efficiency comparison of different methods on the TNO dataset

| Methods            | Time (s) | Parameters (M) | FLOPs (G) |
|--------------------|----------|----------------|-----------|
| GTF      | 3.4207   | /              | /         |
| RFN-Nest  | 2.3096   | 19.17          | 7.68      |
| PMGI  | 0.1934   | 0.04           | 1.69      |
| FusionGAN  | 2.6796 | 0.93           | 0.55      |
| MFEIF   | 0.3181   | 4.94           | 25.30     |
| GANMcC   | 5.6752   | 1.86           | 1.02      |
| PPT Fusion  | 0.4126  | 1.23           | 25.08     |
| DATFuse  | 0.0254  | 0.01           | 1.21      |
| TCCFusion | 0.1220 | 0.19           | 27.08     |
| CrossFuse| 1.0636  | 20.64          | 11.32     |
| MMDRFuse  | 0.0644 | 0.0004         | 0.14      |
| OmniFuse  | 0.0742 | 18.08          | 13.50     |
| PIDFusion | 0.0462 | 0.05          | 208.60    |
| DDBFusion  | 0.9775 | 5.86         | 184.93    |
| DCEvo    | 0.2505   | 2.23           | 2336.42   |
| SETFusion          | 0.2069   | 0.32           | 18.14     |


# Cite the paper
If this work is helpful to you, please cite it as:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@ARTICLE{Tang_2026_SETFusion,
  author={Tang, Wei and He, Fazhi and Zhang, Lin and Zhao, Shengjie },
  journal={Pattern Recognition}, 
  title={SETFusion: A Semantic Transformer for Infrared and Visible Image Fusion}, 
  year={2026},
  volume={},
  number={},
  pages={},
  doi={10.1016/j.patcog.2026.113130}}"><pre class="notranslate"><code>@ARTICLE{Tang_2026_SETFusion,
  author={Tang, Wei and He, Fazhi and Zhang, Lin and Zhao, Shengjie },
  journal={Pattern Recognition}, 
  title={SETFusion: A Semantic Transformer for Infrared and Visible Image Fusion}, 
  year={2026},
  volume={  },
  number={ },
  pages={},
  doi={10.1016/j.patcog.2026.113130}}
</code></pre></div>

If you have any questions,  feel free to contact me (<a href="mailto:weitang@tongji.edu.cn">weitang@tongji.edu.cn</a>).
