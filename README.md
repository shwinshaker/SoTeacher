# SoTeacher

This repo implements teacher regularization methods mentioned in "[Toward Student-Oriented Teacher Network Training For Knowledge Distillation](https://arxiv.org/abs/2206.06661)", including
* Confidence-aware learning (CRL)
* Lipschitz regularization
* Consistency regularization


## Installation
`bash install.sh`


## Running
1. Train teacher networks: `bash run_teacher.sh`
2. Train student networks: `bash run_student.sh`


## Citation
If you find this repo useful, please consider citing the paper

```
@inproceedings{Dong2022SoTeacherAS,
  title={Toward Student-Oriented Teacher Network Training For Knowledge Distillation},
  author={Chengyu Dong and Liyuan Liu and Jingbo Shang},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
For any questions, please contact Chengyu Dong (cdong@ucsd.edu).

## Acknowledgement
This repo is adapted from [RepDistiller](https://github.com/HobbitLong/RepDistiller), which implements a cohort of knowledge distillation methods.