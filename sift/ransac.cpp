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
#include "features2d.hpp"
#include<iostream>
#include <string>
#include<opencv2\opencv.hpp>
#include "colorDetect.h"
#include <stdlib.h>
#include <io.h>
using namespace std;
using namespace cv;

const int BLOCK_NUM = 13;//分块数
const int TOP_NUM = (int)BLOCK_NUM/3;
const int MIN_NUM = 2;
const char* AIM_DIR_PATH = "\\aim";

//RANSAC算法
vector<DMatch> ransac(vector<DMatch> matches,vector<KeyPoint> queryKeyPoint,vector<KeyPoint> trainKeyPoint);

//获取所有的文件名  
void GetAllFilesPath( string path, vector<string>& files)    
{    
  
    long   hFile   =   0;    
    //文件信息    
    struct _finddata_t fileinfo;//用来存储文件信息的结构体    
    string p;    
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  //第一次查找  
    {    
        do    
        {     
            if((fileinfo.attrib &  _A_SUBDIR))  //如果查找到的是文件夹  
            {    
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  //进入文件夹查找  
                {  
                    files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
                    GetAllFilesPath( p.assign(path).append("\\").append(fileinfo.name), files );   
                }  
            }    
            else //如果查找到的不是是文件夹   
            {    
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));  //将文件路径保存，也可以只保存文件名:    p.assign(path).append("\\").append(fileinfo.name)  
            }   
  
        }while(_findnext(hFile, &fileinfo)  == 0);    
  
        _findclose(hFile); //结束查找  
    }   
  
}

int** sift(int** output, cv::Mat img2)
{
	vector<string> files ;
    GetAllFilesPath(AIM_DIR_PATH,files);

	//图像读取
	vector<Mat> img1;//目标小怪
	//Mat img2;
	int k=0;
	for(k =0;k<files.size();k++)
		img1.push_back(imread(files[k],CV_WINDOW_AUTOSIZE));
	//img2=pic;

	if(img1.size()==0||img2.empty())
		return NULL;

	//sift特征提取
	SiftFeatureDetector detector;	
	
	vector<vector<KeyPoint>> keyPoint1;	
	for(k =0;k<files.size();k++){
 		vector<KeyPoint> temp; 
		detector.detect(img1[k],temp);
		keyPoint1.push_back(temp);
	}
	vector<KeyPoint> keyPoint2;
	detector.detect(img2,keyPoint2);
	//sift特征描述子计算
	SiftDescriptorExtractor desExtractor;

	vector<Mat> des1;
	for(k =0;k<files.size();k++){
		Mat temp;
		desExtractor.compute(img1[k],keyPoint1[k],temp);
		des1.push_back(temp);
	}
	Mat des2;
	desExtractor.compute(img2,keyPoint2,des2);

	//sift特征点(描述子)匹配
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	vector<Mat> img_match;
	vector<DMatch> matches_ransac;
	for(k =0;k<files.size();k++){//-----for
	    matcher.match(des1[k],des2,matches);
		matches_ransac=ransac(matches,keyPoint1[k],keyPoint2);
	
	
	Mat img_temp ; 
	drawMatches(img1[k],keyPoint1[k],img2,keyPoint2,matches_ransac,img_temp);
	img_match.push_back(img_temp);

	//imshow(files[k],img_match[k]);
	//ransac(matches[k],keyPoint1[k],keyPoint2);

	//what i do
	int i = 0,j = 0;
	int width = img2.size().width;
	int height = img2.size().height;
	int column_width = width/BLOCK_NUM;
	int raw_height = height/BLOCK_NUM;
	int raw_id = 0,column_id = 0;
	int** temp = new int*[BLOCK_NUM];
	for(i=0;i<BLOCK_NUM;i++){
		temp[i] = new int[BLOCK_NUM];
		for(j=0;j<BLOCK_NUM;j++)
			temp[i][j] = 0;
	}
	for(i=0;i<matches_ransac.size();i++){//将匹配的特征点在原图中标记出来
		int trainIdx = matches_ransac[i].trainIdx;  
		int colunmn_id = (int)keyPoint2[trainIdx].pt.x/column_width;
		int raw_id = (int)keyPoint2[trainIdx].pt.y/raw_height;
		if(raw_id>=0&&raw_id<=2)
			continue;
		if(column_id==BLOCK_NUM)
			column_id-=1;
		if(raw_id==BLOCK_NUM)
			raw_id-=1;
		temp[raw_id][colunmn_id]++;
	}
	for(i=0;i<BLOCK_NUM;i++){
		for(j=0;j<BLOCK_NUM;j++){
			if(temp[i][j]>MIN_NUM)
				output[i][j]=-1;
		}
	}
	}//---------end for
	//what i do

	return output;
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
		if(inliersMask[i])
			matches_ransac.push_back(matches[i]);
	}
	//返回RANSAC过滤后的点对匹配信息
	return matches_ransac;
}

int* getPicOutput(cv::Mat pic)
{
	//##############获得图像识别结果，返回矩阵
	int **output = new int*[BLOCK_NUM];
    int i = 0 ,j = 0;
    for(i = 0;i<BLOCK_NUM;i++){
	    output[i] = new int[BLOCK_NUM];
	    for(j = 0;j<BLOCK_NUM;j++)
		    output[i][j] = 0;
    }
    color_detect(output, pic);
	sift(output, pic);

	int* result = new int[BLOCK_NUM*BLOCK_NUM];
	for(i = 0;i<BLOCK_NUM;i++){
		for(j = 0;j<BLOCK_NUM;j++){
			result[i*BLOCK_NUM+j] = output[i][j];
		    cout<<output[i][j]<<" ";
		}
	   cout<<""<<endl;
   }
	return result;
}


extern int cuteEdge(const cv::Mat& src, cv::Mat& dst);

int main(int argc, char* argv[])
{
	Mat cameraFeed;
	VideoCapture capture;
	Mat img, cropImg;
	double execTime = 0;

	capture.release();
	capture.open(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	
	while(1)
	{
		execTime = (double)getTickCount();
		capture.read(img);

		if(img.empty())
		{
			cout<<"err png:"<<endl;
			waitKey(30);
			continue;
		}

		cuteEdge(img, cropImg);
		//imshow("aaaa", cropImg);
		//waitKey(10000000);
		getPicOutput(cropImg);
		execTime = ((double)getTickCount() - execTime)*1000/getTickFrequency();
		cout<<"exec ms: "<<execTime<<endl;
		waitKey(30);
	}

	waitKey(0);
}


