#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() 
{
	Mat src, src1;
	src = imread("01.png");
	medianBlur(src, src1, 23);
	imshow("input", src);
	int width = src1.cols;
	int height = src1.rows;
	int dims = src1.channels();
	// 初始化定义
	int sampleCount = width * height;
	int clusterCount = 5;//分类数 K
	Mat points(sampleCount, dims, CV_32F, Scalar(10));
	Mat labels;	//表示计算之后各个数据点的最终的分类索引，是一个INT类型的Mat对象
	Mat centers(clusterCount, 1, points.type());
	// 图像RGB到数据集转换
	int index = 0;
	for (int row = 0; row < height; row++) 
	{
		for (int col = 0; col < width; col++) 
		{
			index = row * width + col;
			Vec3b rgb = src1.at<Vec3b>(row, col);
			points.at<float>(index, 0) = static_cast<int>(rgb[0]);
			points.at<float>(index, 1) = static_cast<int>(rgb[1]);
			points.at<float>(index, 2) = static_cast<int>(rgb[2]);
		}
	}
	// 运行K-Means数据分类
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	//criteria表示算法终止的条件，达到最大循环数目或者指定的精度阈值算法就停止继续分类迭代计算
	kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);
	// 显示图像分割结果
	Mat result = Mat::zeros(src1.size(), CV_8UC3);
	for (int row = 0; row < height; row++) 
	{
		for (int col = 0; col < width; col++) 
		{
			index = row * width + col;
			int label = labels.at<int>(index, 0);
			if (label == 1) 
			{
				result.at<Vec3b>(row, col)[0] = 255;
				result.at<Vec3b>(row, col)[1] = 0;
				result.at<Vec3b>(row, col)[2] = 0;
			}
			else if (label == 2) 
			{
				result.at<Vec3b>(row, col)[0] = 0;
				result.at<Vec3b>(row, col)[1] = 255;
				result.at<Vec3b>(row, col)[2] = 0;
			}
			else if (label == 3) 
			{
				result.at<Vec3b>(row, col)[0] = 0;
				result.at<Vec3b>(row, col)[1] = 0;
				result.at<Vec3b>(row, col)[2] = 255;
			}
			else if (label == 0) 
			{
				result.at<Vec3b>(row, col)[0] = 0;
				result.at<Vec3b>(row, col)[1] = 255;
				result.at<Vec3b>(row, col)[2] = 255;
			}
		}
	}
	imshow("kmeans-result", result);

	waitKey(0);
	return 0;
}