/*
 *@function SiftDetect.cpp
 *@brief 对sift特征检测和匹配进行测试，并实现RANSAC算法进行过滤错配点
 *@author ltc
 *@date 11:20 Saturday，28 November，2015
 */
#include "opencv2\highgui\highgui_c.h"
#include "opencv2\highgui\highgui.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv\cv.h"

#include "features2d.hpp"
#include<iostream>
#include<opencv2\opencv.hpp>
using namespace std;
using namespace cv;


//RANSAC算法
vector<DMatch> ransac(vector<DMatch> matches,vector<KeyPoint> queryKeyPoint,vector<KeyPoint> trainKeyPoint);

extern int cuteEdge(const cv::Mat& orglImg, cv::Mat& cropImg);
void cuteEdge2(void);
void cuteEdge3(void);

int main(int argc, char* argv[])
{
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	capture.release();
	capture.open(0);
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	//static const string originalWinName = "Original Image";
	//cv::namedWindow(originalWinName);

	//图像读取
	Mat img1, img2, cropImg;
	img1=imread("D:\\Work\\new\\mario\\sift_monster\\image1.png", CV_WINDOW_AUTOSIZE);
	//img2=imread("D:\\Work\\new\\mario\\sift_monster\\image3.png", CV_WINDOW_AUTOSIZE);
	double execTime = 0;

	while(1)
	{
		execTime = (double)getTickCount();
		capture.read(img2);
		//imshow(cameraFeed);
		if(img1.empty() || img2.empty())
		{
			cout<<"err png:"<<endl;
			waitKey(30);
			continue;
		}
		//imshow("original", img2);
		cuteEdge(img2, cropImg);
		waitKey(30);
	}
	{

		//sift特征提取
		SiftFeatureDetector detector;	
		vector<KeyPoint> keyPoint1,keyPoint2;
		detector.detect(img1,keyPoint1);
		detector.detect(img2,keyPoint2);
		//cout<<"Number of KeyPoint1:"<<keyPoint1.size()<<endl;
		//cout<<"Number of KeyPoint2:"<<keyPoint2.size()<<endl;

		//sift特征描述子计算
		SiftDescriptorExtractor desExtractor;
		Mat des1,des2;
		desExtractor.compute(img1,keyPoint1,des1);
		desExtractor.compute(img2,keyPoint2,des2);

		//sift特征点(描述子)匹配
		Mat res1,res2;
//		drawKeypoints(img1,keyPoint1,res1,Scalar::all(-1),/*DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/DrawMatchesFlags::DEFAULT);
//		drawKeypoints(img2,keyPoint2,res2,Scalar::all(-1),/*DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/DrawMatchesFlags::DEFAULT);
		BruteForceMatcher<L2<float>> matcher;
		FlannBasedMatcher matcher_flann;
		vector<DMatch> matches;

		vector<vector<DMatch>> matches_knn;
		matcher.match(des1,des2,matches);
		matcher.knnMatch(des1,des2,matches_knn,2);

//		cout<<"matches_knn.size:"<<matches_knn.size()<<endl;

		//sift特征最近距离与次近距离之比小于0.6视为正确匹配
		vector<DMatch> match_knn;
		for(int i=0;i<matches_knn.size();i++)
		{
			float ratio=matches_knn[i][0].distance/matches_knn[i][1].distance;
			if(ratio<0.6)
			{
				match_knn.push_back(matches_knn[i][0]);
			}
		}

//		matcher_flann.match(des1,des2,matches_flann);

		//for(int i=0;i<matches.size();i++)
		//{
		//	cout<<"第"<<i<<"对匹配："<<endl;
		//	cout<<"queryIdx:"<<matches[i].queryIdx<<"\ttrainIdx:"<<matches[i].trainIdx<<endl;
		//	cout<<"imgIdx:"<<matches[i].imgIdx<<"\tdistance:"<<matches[i].distance<<endl;
		//}

		//cout<<"Number of matches:"<<matches.size()<<endl;
		//cout<<"Number of matches_flann:"<<matches_flann.size()<<endl;
		
		vector<DMatch> matches_ransac=ransac(matches,keyPoint1,keyPoint2);
		Mat img_match,img_match_flann;
		
		drawMatches(img1,keyPoint1,img2,keyPoint2,matches_ransac,img_match);
		//drawMatches(img1,keyPoint1,img2,keyPoint2,match_knn,img_match_flann);

		//imshow("img_match",img_match); // 连线
		//imshow("img_match_flann",img_match_flann); // 特征值点
		
		//for(size_t i=0;i<keyPoint1.size();i++)
		//{
		//	//cout<<"x:"<<kp1.at(i).pt.x<<endl;
		//	circle(img1,Point((int)(keyPoint1.at(i).pt.x),(int)(keyPoint1.at(i).pt.y)),3,Scalar(255,0,0),2,8,0);
		//}

		//imshow("img1",img1);
		//imshow("img1_",res1);
//		imshow("img2",res2);
		ransac(matches,keyPoint1,keyPoint2);
		execTime = ((double)getTickCount() - execTime)*1000/getTickFrequency();
		cout<<"exec ms: "<<execTime<<endl;
		waitKey(30);
	}

	return 0;
}


//RANSAC算法实现
vector<DMatch> ransac(vector<DMatch> matches,vector<KeyPoint> queryKeyPoint,vector<KeyPoint> trainKeyPoint)
{
	//定义保存匹配点对坐标
	vector<Point2f> srcPoints(matches.size()),dstPoints(matches.size());
	//保存从关键点中提取到的匹配点对的坐标
	for(int i=0;i<matches.size();i++)
	{
		srcPoints[i]=queryKeyPoint[matches[i].queryIdx].pt;
		dstPoints[i]=trainKeyPoint[matches[i].trainIdx].pt;
	}
	//保存计算的单应性矩阵
	Mat homography;
	//保存点对是否保留的标志
	vector<unsigned char> inliersMask(srcPoints.size()); 
	//匹配点对进行RANSAC过滤
	homography = findHomography(srcPoints,dstPoints,CV_RANSAC,5,inliersMask);
	//RANSAC过滤后的点对匹配信息
	vector<DMatch> matches_ransac;
	//手动的保留RANSAC过滤后的匹配点对
	for(int i=0;i<inliersMask.size();i++)
	{
		//cout<<inliersMask[i]<<endl;
		cout<<(int)(inliersMask[i])<<endl;
		if(inliersMask[i])
		{
			matches_ransac.push_back(matches[i]);
			//cout<<"第"<<i<<"对匹配："<<endl;
			//cout<<"queryIdx:"<<matches[i].queryIdx<<"\ttrainIdx:"<<matches[i].trainIdx<<endl;
			//cout<<"imgIdx:"<<matches[i].imgIdx<<"\tdistance:"<<matches[i].distance<<endl;
		}
	}
	//返回RANSAC过滤后的点对匹配信息
	return matches_ransac;
}


void cuteEdge2(void)
{

	// 去除黑边
	// Convert RGB Mat to GRAY
	Mat img2 = imread("D:\\Work\\new\\mario\\sift_monster\\68.jpeg", CV_WINDOW_AUTOSIZE);
	Mat gray2;
	cv::Mat gray;
	cv::cvtColor(img2, gray, CV_BGR2GRAY);

	// Store the set of points in the image before assembling the bounding box
	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = gray.begin<uchar>();
	cv::Mat_<uchar>::iterator end = gray.end<uchar>();
	for (; it != end; ++it)
	{
		if (*it > 50) 
		{
			//cout<<" "<<*it;
			points.push_back(it.pos());
		}
	}
	//while(1);

	// Compute minimal bounding box
	cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

	// Draw bounding box in the original image (debug purposes)
	//cv::Point2f vertices[4];
	//box.points(vertices);
	//for (int i = 0; i < 4; ++i)
	//{
	//cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, CV_AA);
	//}
	//cv::imshow("box", img);
	//cv::imwrite("box.png", img);

	// Set Region of Interest to the area defined by the box
	cv::Rect roi;
	roi.x = box.center.x - (box.size.width / 2);
	roi.y = box.center.y - (box.size.height / 2);
	roi.width = box.size.width;
	roi.height = box.size.height;
	cout<<"box: "<<box.center.x<<" "<<box.center.y<<" "<<box.size.width<<" "<<box.size.height<<endl;
	cout<<"roi: "<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height<<endl;
	while(1);
	// Crop the original image to the defined ROI
	//cv::Mat crop = img2(roi);
	//cv::imshow("crop", crop);

	//cv::imwrite("cropped.png", crop);
	waitKey(10000000);
}


