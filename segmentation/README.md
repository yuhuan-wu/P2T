## [TPAMI22] Pyramid Pooling Transformer for Scene Understanding

This folder contains full training and test code for semantic segmentation.

### Requirements

* mmsegmentation >= 0.12+

### Results (val set) & Pretrained Models)


|  Base Model    | Variants  | mIoU | aAcc | mAcc | #Params (M) | # GFLOPS |                         Google Drive                         |
| :--: | :-------: | :--: | :--: | :---------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Semantic FPN    | P2T-Tiny  | 43.4 | 80.8 |    54.5    |    15.4    |   31.6   | [[weights]](https://drive.google.com/file/d/1x9EweWx77pXrHOCc7RJF3sYK2rht0_4m/view?usp=sharing)\|[[log]](https://drive.google.com/file/d/1CDofCg9pi0Cyiha_dIimggF228M5mOeH/view?usp=sharing) |
| Semantic FPN    | P2T-Small | 46.7 | 82.0 |    58.4    |    27.8    |   42.7   | [[weights]](https://drive.google.com/file/d/1FlwhyVKw0zqj2mux248gIQFQ8DGPi8rS/view?usp=sharing)\|[[log]](https://drive.google.com/file/d/1bCZz7y0I0EEw74KaVg5iAr3hBYtSIEii/view?usp=sharing) |
| Semantic FPN    | P2T-Base  | 48.7 | 82.9 |    60.7    |    39.8    |   58.5   | [[weights]](https://drive.google.com/file/d/1iZoWexUTPUDSIZiJHNRt2zZl2kFj68F4/view?usp=sharing)\|[[log]](https://drive.google.com/file/d/13_XaX0XtYSzPatVl54ihFbEwflHLVvsl/view?usp=sharing) |
| Semantic FPN      | P2T-Large | 49.4 | 83.3 |    61.9    |    58.1    |   77.7   | [[weights]](https://drive.google.com/file/d/13jBJ7ShDJd1juViC-zPtfLXYPRwkNfya/view?usp=sharing)\|[[log]](https://drive.google.com/file/d/1-RLjGzez-_O2_8obbXvUYGhWacPnqK1U/view?usp=sharing) |



### Train

Use the following commands to train `Semantic FPN` with `P2T-Small` backbone for distributed learning with 8 GPUs:

````
bash dist_train.sh configs/sem_fpn_p2t_s_ade20k_80k.py 8
````

### Validate

Please download the pretrained model from the above table. Put them to `pretrained` folder.
Then, use the following commands to validate `Semantic FPN` with `P2T-Small` backbone in a single GPU:

````
bash dist_test.sh configs/sem_fpn_p2t_s_ade20k_80k.py pretrained/sem_fpn_p2t_s_ade20k_80k.pth 1
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
}
````

### License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public Lincense for Non-Commercial use only. Any commercial use should get formal permission first.

