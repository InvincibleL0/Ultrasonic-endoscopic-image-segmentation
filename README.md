# Ultrasonic-endoscopic-image-segmentation
一、项目简介

	超声内窥成像是基于检测超声信号在组织中的回波进行成像，可以对组织层次及附近器官成像，反映组织声阻抗的差异性。无创、无辐射、实时，从而可用于检查深层
信息。超声内窥成像可以识别肠壁的组织结构，直观的分辨出各层的差异和界限。肠道疾病的早期诊断对于治疗具有重要意义，早期的肠道疾病往往伴随肠壁水肿、增厚，
肠壁由黏膜、黏膜下层、肌层和浆膜层构成，肠壁增厚是肠道疾病的主要超声相关指标之一，不同层次的组织结构增厚可能反映了不同的肠道疾病类型。
	对肠壁形态定量评估需要在肠壁横截面图像中将不同层次的组织结构的界限准确地描绘出来。然而单一横截面图像的定量评估对于诊断是无意义的，只有对大量连续的
横截面组成的3D图像进行定量分析才能获得组织结构形态信息。当进入3D分割时，由于需要描绘大量2D成像切片，这种人工描绘边界变得十分耗时，自动化的肠壁分割方法
使得定量分析更加现实，进而促进肠壁超声图像定量分析在临床中的应用。（fig.1）显示了肠壁横截面的组织结构。超声内窥图像目标区域灰度值范围有较大重叠，噪声
较明显，虽然人工标注界限比较轻松，但由于超声图像组织结构的非均匀性，传统的非机器学习图像处理方法不能很好的将不同的声阻抗层识别并分割（fig.2）。
	随着可用于图像分割的卷积神经网络的兴起。基于监督学习的端到端卷积神经网络方法，在此分割任务中可以发挥作用。目前，基于深度学习的图像分割方法在医学图
像领域的应用越来越广泛，例如眼底图像血管分割、肺部CT图像分割、血管内超声图像分割等等。为了解决这一计算机视觉问题，我们提出了一种类似于Unet的全卷积神经
网络方法，相较于传统图像分割技术，基于全卷积神经网络的方法，有更高的分割准确率和更低的时间复杂度。并且，基于全神经网络的方法可以做到多分类，将各层次的
组织结构区分开来，省去了繁杂的区域选取步骤。

![FIG1](https://user-images.githubusercontent.com/55590536/111862261-2b9d7480-898f-11eb-8d31-d19229b6751a.png)

二、实现方法

	我们用之前开发的超声内窥成像系统对离体猪肠进行了成像，为分割任务准备一个数据集，提供给网络进行训练和测试。成像系统超声频率为40Mhz，由（fig.1.A和B）
所示，肠壁由内而外呈现“暗-亮-暗-亮”的四层结构，分别为黏膜层、黏膜下层、肌层和浆膜层。数据集包含500张图像，分为训练集、验证集和测试集，比例为8：1：1。
	如（fig.1.D）所示，我们开发了一种结构类似于U型的全卷积神经网络，主要由编码区和解码区组成。编码区包含4次下采样以获得高级特征映射，对应的解码区包含4
次上采样以恢复图像原始分辨率并获得每个像素点的分类概率。在4次下采样前和4次上采样后添加了对应的合并结构，目的是结合高低语义信息，获得更细致的特征映射。在
此全卷积神经网络中，编码区5个卷积单元、解码区4个卷积单元共19个卷积层、4个最大值汇聚层、4个上采样层。其中18个卷积层由大小3×3的滑动窗口卷积滤波器、线性修
正单元组成，用来进行特征提取。最末尾的卷积层由大小1×1的滑动窗口卷积滤波器和Softmax激活函数组成，用来获得概率分布。最大值汇聚层为大小为2×2的滑动窗口滤波
器，获得每个2×2像素区域的最大值，达到下采样的目的，使得特征图大小变为原来的四分之一，可以实现参数减少和特征选取。4个上采样层为2×2的滑动窗口卷积滤波器，
插值方式为双线性插值，使得特征图大小扩大四倍，通过4个这样的上采样层将特征图恢复到原始图像大小。此外，为了避免过拟合，我们在最低分辨率的特征层使用了
Dropout结构。
	卷积神经网络的输入为512×512×1的2D超声肠壁图像切片，进入编码区第一层卷积单元，64个参数可学习的卷积滤波器分别与图像进行卷积，生成64个特征映射，特征映
射的大小与输入图像一致，最大值池化层将图像下采样到256×256。随后的卷积单元继续生成图像特征映射，并减小图像大小，由此实现低级和高级卷积特征的分层提取。最底
层卷积单元，生成1024个特征映射，将图像下采样到32×32。
	上述编码区生成的特征映射进入解码区，经过参数可学习的上采样卷积层将图像大小增大4倍，随后与编码区中相应的单元进行合并，以更好的利用编码区中的特征映射恢
复图像细节信息，并通过和编码区相同设置的卷积单元。经过4次上采样后，图像大小恢复到与输入图像大小一致。最后，经过一个包含3个大小1×1的卷积滤波器层降图像降至
3维，激活函数为Softmax，生成每个像素点的概率分布。卷积神经网络的损失函数为交叉熵函数，作为网络卷积滤波器参数学习的依据。利用DICE系数来评价卷积神经网络的
分割结果与人工标注真实分割结果之间的重合率。
	该神经网络用python编程语言和tensorflow深度学习框架实现，学习率为0.0001。
	
	![FIG2](https://user-images.githubusercontent.com/55590536/111862351-c7c77b80-898f-11eb-9e0f-66d6e475ae0d.png)

