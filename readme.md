# 期中作业

### 1. 使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。
    - 配置：在./configs/config_task1.py中可以设置实验的参数、数据增强的方式。
    - 训练：运行./task1_train.py，训练好的模型将被保存到./trained_model/*中。
    - 测试：运行./task1_test.py，调用训练好的模型在测试集上检验模型效果。
    - 实验结果：实验结果存储在./task1_result.md中。
    - 可视化：运行./task1_visualization.ipynb，生成三种数据增强方法的图像可视化结果。

### 2. 在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；两个训练好后的模型分别可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并进行对比，一共show六张图像；
    - 在VOC2007数据集官方网站下载VOC2007数据集，并保存在dataset/VOCdevkit中
    - 运行 task2_train_fasterrcnn.py在VOC2007数据集上训练faster_rcnn，训好的模型将保存在trained_model中。
    - 运行 task2_inference_fasterrcnn.py, 并设置 48行 VOC=True, 49行 drawproposal= True, 能够可视化testset中图像在模型第一阶段输出的proposal_box，并保存在imgs/task2_output中。
    - 运行 task2_inference_fasterrcnn.py，并设置48行VOC=False，能够可视化imgs/task2/input文件夹下的几张不在VOC2007数据集中的图像的物体检测结果，结果图像保存在 imgs/task2/output中。
    - 运行 task2_train_fcos.py在VOC2007数据集上训练fcos，训好的模型将保存在checkpoint/model.pth中。
    - 运行 task2_inference_fcos.py，可视化imgs/task2/input文件夹下的几张不在VOC2007数据集中的图像的物体检测结果，结果图像保存在 imgs/task2/output中。

各训练好的模型网盘地址为：
