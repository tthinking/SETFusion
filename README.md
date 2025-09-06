
SETFusion is now under the second-round revision. The source code and trained model will be released once accepted.

### Visualizations of the feature maps from the proposed PSTM and MSTM modules.
![Image](figs/featureMaps.png)

 <!--
## Computational efficiency comparison of different methods on TNO dataset


| Method | Time (s) | Parameters (M) |
| :---: | :---: | :---: |
| GTF | 3.4207 | / |
|RFN-Nest|	2.3096 |	19.17 |
|PMGI|	0.1934 |	0.04 |
|FusionGAN|		2.6796 |	0.93 |
|MFEIF|	0.3181 |	4.94|
|GANMcC|	5.6752|	1.86 |
|PPT Fusion|	0.4126|1.23 |
|DATFuse|	0.0254 	|0.01 |
|TCCFusion|		0.1220 |0.19 |
|CrossFuse |	1.0636 |	20.64 |
|SETFusion|	0.2069 |	0.32|
-->

### Comparison on the downstream task, i.e., object detection.
![Image](figs/objectDetection.png)

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
