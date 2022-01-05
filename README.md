## Pyramid Pooling Transformer for Scene Understanding

Requirements:

* torch 1.6+
* torchvision 0.7.0
* `timm==0.3.2`
* Validated on torch 1.6.0, torchvision 0.7.0

### Models Pretrained on ImageNet1K

|     Variants     | Input  Size    | Acc@1 | Acc@5 | #Params (M) | Pretrained Models |
|-----------------|:---------:|:-----:|:-----:|:-----------:|:-----------------:|
| P2T-Tiny   | 224 x 224 |  78.1 |  94.1 |     11.1    |     [Google Drive](https://drive.google.com/file/d/181mr1zbCtJFQZinbxO70v59-p6gLLCoM/view?usp=sharing)    |
| P2T-Small  | 224 x 224 |  82.1 |  95.9 |     23.0    |     [Google Drive](https://drive.google.com/file/d/1_UUrrQdbAiNf4y5gZCETG23h7EYMV18O/view?usp=sharing)    |
| P2T-Base | 224 x 224 |  83.0 |  96.2 |     36.2    |     [Google Drive](https://drive.google.com/file/d/1fEIMyT05a0Oj6tPdNaVxITJugg4sQKAo/view?usp=sharing)    |

### Pretrained Models for Downstream tasks

To be updated.

### Something Else
Note: we have prepared a stronger version of P2T. Since P2T is still in peer review, we will release the stronger P2T after the acceptance.
