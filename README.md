# HybridPillars: Hybrid Point-Pillar Network for Real-time Two-stage 3D Object Detection


Abstract: LiDAR-based 3D object detection is an important perceptual task in various fields such as intelligent transportation, autonomous driving and robotics. Existing two-stage point-voxel methods contribute to the boost of accuracy on 3D object detection by utilizing precise point-wise features to refine 3D proposals. Although obtaining promising results, these methods are not suitable for real-time applications. Firstly, the inference speed of existing point-voxel hybrid frameworks is slow because the acquisition of point features from voxel features consumes a lot of time. Secondly, existing point-voxel methods rely on 3D convolution for voxel feature learning, which increases the difficulty of deployment on embedded computing platforms. To address these issues, we propose a real-time two-stage detection network, named HybridPillars.
We first propose a novel hybrid framework by integrating a point feature encoder into a point-pillar pipeline efficiently. By combining point-based and pillar-based networks, our method can discard 3D convolution to reduce computational complexity. Furthermore, we propose a novel pillar feature aggregation network to efficiently extract BEV features from point-wise features, thereby significantly enhancing the performance of our network. Extensive experiments demonstrate that our proposed HybridPillars not only boosts the inference speed, but also achieves competitive detection performance compared with other methods. 

<p align="center"> <img src="docs/network.jpg" width="100%"> </p>
## 1. Recommended Environment
- Ubuntu 18.04
- Python 3.7.13
- PyTorch 1.7.0, cuda 11.0 version
- CUDA NVCC 11.1
- Spconv 2.1.21

## 2. Set the Environment
``` bash
pip install -r requirement.txt
bash compile.sh
```
## 3. Prepare Data
- Prepare KITTI dataset and road planes
```bash
# Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

# Generatedata infos:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## 4. Train
```
cd tools
python ./train.py --cfg_file ./cfg/kitti_models/hybridpillars.yaml

or 

bash tools/scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```
## 5. Test
```
python test.py --cfg-file ${CONFIG_FILE} --ckpt ${CKPT}
```
## 6. Acknowledgement
- Thanks for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), this implementation is mainly based on the pcdet v0.6.0.
- Parts of our code refer to the excellent work [IA-SSD](https://github.com/yifanzhang713/IA-SSD).

## 6. Citation
