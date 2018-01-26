## 迁移学习
参考官网例子，实现ResNet18（ImageNet数据集训练）的网络模型进行迁移分类两种动物
- ./data/ 数据集文件夹里面存放ants和bees的训练集和验证集
- transform_bees_ants.py 迁移学习（方法一）将最后的全连接层改变输出来训练网络
- transform_extractor.py 迁移学习（方法二）将网络作为特征提取器参数不动 
