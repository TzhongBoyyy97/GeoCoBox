# GeoCoBox: Box-supervised 3D Tumor Segmentation via Geometric Co-embedding

This paper has been accepted for presentation at AAAI 2026 as an Oral Presentation. The preprint can be found at: [AAAI_4465_GeoCoBox.pdf](https://github.com/user-attachments/files/23772848/AAAI_4465_GeoCoBox.pdf)
. And our code is based on [DRMNet](https://github.com/TzhongBoyyy97/DRMNet).

This code is licensed for non-commercial research purposes only.

## Contributions

We propose a 3D tumor segmentation model for box supervision, namely GeoCoBox. It explicitly embeds the positional information with contrastive features, enabling intertask collaboration.

We introduce an Anatomical-Driven Class Activation Map for predefining the tumor morphology that can provide an anatomical prior for pixel-wise learning.

We propose a Geometric Pixel Co-embedding Learner for refining the tumor boundaries. This method, along with our proposed contrastive head pretraining strategy, effectively utilizes the tumor center information, thereby reducing computational overhead. 

## Motivation

The current box-supervised segmentation pipelines. The comparison between (a) pseudo-label generation models, (b) loss-based models, (c) organ template learning models, and (d) our approach.
<img width="1889" height="1453" alt="fig1-relawork" src="https://github.com/user-attachments/assets/71ade667-4cc8-4801-83af-5f080c405c3b" />


## Method
The GeoCoBox framework uses 3D-UNet as the backbone. It includes two branches: AD-CAM and GCL. The AD-CAM branch generates coarse masks and provides edge positions. The GCL branch is employed to compute the similarity between the center embedding and the edge embeddings provided by the pre-trained contrastive head. The red line indicates that GCL explicitly integrates the positional information provided by AD-CAM and the embedding information from the contrastive head. 
<img width="2236" height="1262" alt="fig2-newmethod" src="https://github.com/user-attachments/assets/748747d0-7c60-4db5-b822-6d28ef26b421" />


## Citations
If you are using the code/model/data provided here in a publication, please consider citing:
```bibtex
@inproceedings{lan2025loobox,
  title={LooBox: Loose-box-supervised 3D Tumor Segmentation with Self-correcting Bidirectional Learning},
  author={Lan, Tianzhong and Yi, Zhang and Xu, Xiuyuan and Zhu, Min},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={8077--8086},
  year={2025}
}
```

## Contact
For any questions, please contact me via e-mail: lantianzhong1@stu.scu.edu.cn.
