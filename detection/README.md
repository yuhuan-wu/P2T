## [TPAMI22] Pyramid Pooling Transformer for Scene Understanding

This folder contains full training and test code for semantic segmentation.

### Requirements

* mmdetection == 2.14

We train each model based on `mmdetection==2.8.0`.
Since new GPU cards (RTX 3000 series) should compile mmcv from source to support this early version,
we reorganize the config and support newer mmdetection version. 
Therefore, you can simply reproduce the result on newer GPUs.

### Data Preparation

Put MS COCO dataset files to `data/coco/`.

### Object Detection

Tested on the coco validation set


|  Base Model    | Variants  | AP | AP@0.5 | AP@0.75 | #Params (M) | # GFLOPS |
| :--: | :-------: | :--: | :--: | :---------: | :------: | :----------------------------------------------------------: |
| RetinaNet    | P2T-Tiny  | 41.3 | 62.0 |    44.1    |    21.1    |   206   |
| RetinaNet  | P2T-Small | 44.4 | 65.3 |    47.6    |    33.8    |   260   |
| RetinaNet  | P2T-Base  | 46.1 | 67.5 |    49.6    |    45.8    |   344    |
| RetinaNet  | P2T-Large | 47.2 | 68.4 |    50.9    |    64.4    |   449   |

Use this address to access all pretrained weights and logs: [[Google Drive]](https://drive.google.com/drive/folders/1fcg7n3Ga8cYoT-3Ar0PeQXjAC3AnQYyY?usp=sharing)

### Instance Segmentation 

Tested on the coco val set


|  Base Model    | Variants  | APb | APb@0.5 | APm  | APm@0.5 | #Params (M) | # GFLOPS |
| :--: | :-------: | :--: | :--: | :---------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Mask R-CNN | P2T-Tiny  | 43.3 | 65.7 |    39.6    |    62.5    |    31.3     |   225   |
| Mask R-CNN | P2T-Small | 45.5 | 67.7 |    41.4    |    64.6    |    43.7     |   279   |
| Mask R-CNN | P2T-Base  | 47.2 | 69.3 |    42.7    |    66.1    |    55.7    |   363   |
| Mask R-CNN | P2T-Large | 48.3 | 70.2 | 43.5 |    67.3    |    74.0    |   467   |

`APb` denotes AP box metric, and `APm` is the AP mask metric.

Use this address to access all pretrained weights and logs: [[Google Drive]](https://drive.google.com/drive/folders/1fcg7n3Ga8cYoT-3Ar0PeQXjAC3AnQYyY?usp=sharing)


### Train

Before training, please make sure you have `mmdetection==2.14` and downloaded the ImageNet-pretrained P2T weights from [[Google Drive]](https://drive.google.com/drive/folders/1Osweqc1OphwtWONXIgD20q9_I2arT9yz?usp=sharing) or
[[BaiduPan, 提取码yhwu]](https://pan.baidu.com/s/1JkE62CS9EoSTLW1M1Ajmxw?pwd=yhwu). 
Put them to `pretrained/` folder.

Use the following commands to train `Mask R-CNN` with `P2T-Tiny` backbone for distributed learning with 8 GPUs:

````
bash dist_train.sh configs/mask_rcnn_p2t_t_fpn_1x_coco.py 8
````

Other configs are on the `configs` directory.

### Validate

Please download the pretrained model from [[Google Drive]](https://drive.google.com/drive/folders/1fcg7n3Ga8cYoT-3Ar0PeQXjAC3AnQYyY?usp=sharing) or [[BaiduPan, 提取码yhwu]](https://pan.baidu.com/s/1JkE62CS9EoSTLW1M1Ajmxw?pwd=yhwu). Put them to `pretrained` folder.
Then, use the following commands to validate `Semantic FPN` with `P2T-Small` backbone in a single GPU:

````
bash dist_test.sh configs/mask_rcnn_p2t_t_fpn_1x_coco.py pretrained/mask_rcnn_p2t_t_fpn_1x_coco-d875fa68.pth 1
````


### Other Notes

If you meet any problems, please do not hesitate to contact us.
Issues and discussions are welcome in the repository!
You can also contact us via sending messages to this email: wuyuhuan@mail.nankai.edu.cn



### Citation

If you are using the code/model/data provided here in a publication, please consider citing our works:

````
@ARTICLE{wu2022p2t,
  author={Wu, Yu-Huan and Liu, Yun and Zhan, Xin and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{P2T}: Pyramid Pooling Transformer for Scene Understanding}, 
  year={2022},
  doi = {10.1109/tpami.2022.3202765},
}
````

### License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Non-Commercial use only. Any commercial use should get formal permission first.

