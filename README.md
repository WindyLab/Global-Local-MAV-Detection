<div align="center">
  <h1>Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2312.11008">
    <img src="https://img.shields.io/badge/arXiv-paper?style=socia&logo=arxiv&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://arxiv.org/pdf/2312.11008.pdf">
    <img src="https://img.shields.io/badge/Paper-blue?logo=googledocs&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://westlakeu-my.sharepoint.com/:f:/g/personal/zhao_lab_westlake_edu_cn/EgX-57n5etFOtaS_QjeGfQEBOTo6y9PkVOGTyt3tsOB5LA?e=jILuMf">
    <img src="https://img.shields.io/badge/Dataset-blue?logo=microsoftsharepoint&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://pan.baidu.com/share/init?surl=qROfavqy_auzfq0mqjiJ3A?pw=sr7f">
    <img src="https://img.shields.io/badge/Baidu Netdisk-blue?logo=dask&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://www.youtube.com/watch?v=Tv473mAzHbU">
    <img src="https://img.shields.io/badge/Video-blue?logo=youtube&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>
  
  [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Tv473mAzHbU/0.jpg)](https://www.youtube.com/watch?v=Tv473mAzHbU)
  
</div>

This is the repository for the paper "Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion". This paper is conditionally accepted by IEEE Transactions on Intelligent Transportation Systems. 

In this paper, we propose a global-local MAV detector that can fuse both motion and appearance features for MAV detection under challenging conditions.

![architecture](https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection/assets/125523389/656b737a-d846-4206-9d9b-0a4faec524af)

### Dataset
- [BaiduYun](https://pan.baidu.com/share/init?surl=qROfavqy_auzfq0mqjiJ3A?pw=sr7f)
- [GoogleDrive](https://drive.google.com/file/d/1_I5jR-a-Jlan96s7XD3QeLLddb51rDT_/view?usp=drive_link)
%- [Microsoft SharePoint](https://westlakeu-my.sharepoint.com/:f:/g/personal/zhao_lab_westlake_edu_cn/EgX-57n5etFOtaS_QjeGfQEBOTo6y9PkVOGTyt3tsOB5LA?e=jILuMf)
  
This dataset includes 60 videos andd 106665 frames, the average object size is only 0.02% of the image size (1920 * 1080). The annotation files follow the Pascal VOC XML format. In addition, we provide a python code for extracting images from a video.

In this paper, 45 videos are used for model trainning, and 15 videos are used for testing. The video ID for testing is:05, 08, 09, 10, 19, 30, 41, 43, 46, 47, 58, 63,  65, 70, 86.

If you have any problem when using this dataset, please feel free to contact: guohanqing@westlake.edu.cn.

## Citing
If you find our work useful, please consider citing:
```BibTeX
@misc{guo2023globallocal,
      title={Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion}, 
      author={Hanqing Guo and Ye Zheng and Yin Zhang and Zhi Gao and Shiyu Zhao},
      year={2023},
      eprint={2312.11008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
