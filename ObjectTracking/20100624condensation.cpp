//condensation.cpp
//クリックした部分の色を追跡します。


#include <Windows.h>
#include <stdio.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\legacy\legacy.hpp>
#include <ctype.h>
#include <math.h>

int filterhue;
float sigma =40;

//クリックした点の座標
float initx = 0;
float inity = 0;
//クリックした点のRGB
unsigned char blue = 0;
unsigned char red = 0;
unsigned char green = 0;
//クリックした座標のHSV
int h = 0;
int s = 0;
int v = 0;
CvScalar targetColor;
IplImage *frame = 0;
IplImage *redimage=0;
IplImage *greenimage=0;
IplImage *blueimage=0;

cv::Mat capframe;
//閾値
int thresh = 10;


//RGB->HSVの変換
void RGBtoHSV(int r, int g, int b, int* h, int* s, int* v)
{
	int max = 0;
	if(r >= g && r >= b) max = r;
	else if(g >= r && g >= b) max = g;
	else max = b;
	int min = 0;
	if(r <= g && r <= b) min = r;
	else if(g <= r && g <= b) min = g;
	else min = b;
	//V
	*v = max;
	if(max == 0 || max == min) {
		*s = 0; *v = 0;
	}else{
		//S
		*s = 255 * (max - min) / max;
		//H
		if(max == r) *h = 60 * (b - g) / (max - min);
		else if(max == g) *h = 60 * (2 + (r - b) / (max - min));
		else *h = 60 * (4 + (g - r) / (max - min));
	}
}

//尤度(もっともらしさ)の計算を行なう関数
float calc_likelihood (IplImage * img, int x, int y)
{
		float dist = 0.0;
    unsigned char b, g, r;
    b = img->imageData[img->widthStep*y + x*3];     // B
    g = img->imageData[img->widthStep*y + x*3 + 1]; // G
    r = img->imageData[img->widthStep*y + x*3 + 2]; // R
    
		
 	 dist = sqrt((float)(blue-b)*(blue-b) +(float)(green-g)*(green-g) +(float)(red-r)*(red-r));
   
	 // 距離(dist)を平均、sigmaを分散として持つ、正規分布を尤度関数とする
   return 1.0 / (sqrt(2.0*CV_PI)*sigma) * expf(-dist*dist/(2.0*sigma*sigma)) ;
}

//particle filter用
void on_mouse( int event, int x, int y, int flags, void* param )
{
    
    if( event == CV_EVENT_LBUTTONDOWN )
    {
		  if(frame->nChannels==3)
		  {
		 		int r,g,b,count;
				r=g=b=0;
				count=0;
  			for(int i=-3;i<3;i++)
				{			
					for(int j=-3;j<3;j++)
					{
						b =b+ frame->imageData[frame->widthStep*(y+j) +(x+i)*3];     // B
						g =g+ frame->imageData[frame->widthStep*(y+j) +(x+i)*3 + 1]; // G
						r =r+ frame->imageData[frame->widthStep*(y+j) +(x+i)*3 + 2]; // R
						count++;
					}
				}
				
				//平均値を検索対象にする
				blue=b/count;
				green=g/count;
				red=r/count;
			}
    }
}

void _onMouse( int event, int x, int y, int flags, void* param )
{
    initx = (float)x; 
	inity = (float)y;

    if( event == CV_EVENT_LBUTTONDOWN )
    {
		int r,g,b,count;
		int sumR, sumG, sumB;
		r = g = b = sumR = sumG = sumB = 0;
		count=0;

		int sumH, sumS, sumV;
		int _h, _s, _v;
		_h = _s = _v = sumH = sumS = sumV = 0;

  		for(int i=-3;i<3;i++)
			{			
				for(int j=-3;j<3;j++)
				{
					b = capframe.at<cv::Vec3b>(y+j, x+i)[0];
					g = capframe.at<cv::Vec3b>(y+j, x+i)[1];
					r = capframe.at<cv::Vec3b>(y+j, x+i)[2];
					sumR += r;
					sumG += g;
					sumB += b;

					RGBtoHSV(r, g, b, &_h, &_s, &_v);

					sumH += _h;
					sumS += _s;
					sumV += _v;

					count++;
				}
			}

				h = sumH / count;
				s = sumS / count;
				v = sumV / count;
				
				//平均値を検索対象にする
				blue = sumB/count;
				green = sumG/count;
				red = sumR/count;
    }
}

//particle filter
//int main (int argc, char **argv)
//{
//	int i, c;
//  double w = 0.0, h = 0.0;
//  CvCapture *capture = 0;
//
//  int n_stat = 4;
//  int n_particle = 4000;
//  CvConDensation *cond = 0;
//  CvMat *lowerBound = 0;
//  CvMat *upperBound = 0;
//  int xx, yy;
//
//	capture = cvCreateCameraCapture (0);
//
//  //１フレームキャプチャし，キャプチャサイズを取得する．
//	frame = cvQueryFrame (capture);
// 	redimage=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
//	greenimage=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
//	blueimage=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
//	w = frame->width;
//    h = frame->height;
//	cvNamedWindow ("Condensation", CV_WINDOW_AUTOSIZE);
//	cvSetMouseCallback("Condensation",on_mouse,0);
// 
// 	//フォントの設定
//	CvFont dfont;
//	float hscale      = 0.7f;
//	float vscale      = 0.7f;
//	float italicscale = 0.0f;
//	int  thickness    = 1;
//	char text[255] = "";
//	cvInitFont (&dfont, CV_FONT_HERSHEY_SIMPLEX , hscale, vscale, italicscale, thickness, CV_AA); 
//
//	//Condensation構造体を作成する．
//	cond = cvCreateConDensation (n_stat, 0, n_particle);
//
//
//  //状態ベクトル各次元の取りうる最小値・最大値を指定する
//	//今回は位置(x,y)と速度(xpixcel/frame,ypixcel/frame)の4次元
//
//	lowerBound = cvCreateMat (4, 1, CV_32FC1);
//  upperBound = cvCreateMat (4, 1, CV_32FC1);
//  cvmSet (lowerBound, 0, 0, 0.0);
//  cvmSet (lowerBound, 1, 0, 0.0);
//  cvmSet (lowerBound, 2, 0, -20.0);
//  cvmSet (lowerBound, 3, 0, -20.0);
//  cvmSet (upperBound, 0, 0, w);
//  cvmSet (upperBound, 1, 0, h);
//  cvmSet (upperBound, 2, 0, 20.0);
//  cvmSet (upperBound, 3, 0, 20.0);
//  
//	//Condensation構造体を初期化する
//	cvConDensInitSampleSet (cond, lowerBound, upperBound);
//
//  //ConDensationアルゴリズムにおける状態ベクトルのダイナミクスを指定する
//	cond->DynamMatr[0] = 1.0;
//  cond->DynamMatr[1] = 0.0;
//  cond->DynamMatr[2] = 1.0;
//  cond->DynamMatr[3] = 0.0;
//  cond->DynamMatr[4] = 0.0;
//  cond->DynamMatr[5] = 1.0;
//  cond->DynamMatr[6] = 0.0;
//  cond->DynamMatr[7] = 1.0;
//  cond->DynamMatr[8] = 0.0;
//  cond->DynamMatr[9] = 0.0;
//  cond->DynamMatr[10] = 1.0;
//  cond->DynamMatr[11] = 0.0;
//  cond->DynamMatr[12] = 0.0;
//  cond->DynamMatr[13] = 0.0;
//  cond->DynamMatr[14] = 0.0;
//  cond->DynamMatr[15] = 1.0;
//  
//	//ノイズパラメータを再設定する．
//	cvRandInit (&(cond->RandS[0]), -25, 25, (int) cvGetTickCount ());
//  cvRandInit (&(cond->RandS[1]), -25, 25, (int) cvGetTickCount ());
//  cvRandInit (&(cond->RandS[2]), -5, 5, (int) cvGetTickCount ());
//  cvRandInit (&(cond->RandS[3]), -5, 5, (int) cvGetTickCount ());
//  while (1) 
//	{
//		frame = cvQueryFrame (capture);
//
//
//		//各パーティクルについて尤度を計算する．
//		for (i = 0; i < n_particle; i++)
//		{ 
//			xx = (int) (cond->flSamples[i][0]);
//			yy = (int) (cond->flSamples[i][1]);
//			 if (xx < 0 || xx >= w || yy < 0 || yy >= h) 
//				{  
//					cond->flConfidence[i] = 0.0;
//				}
//				else
//				{  
//					cond->flConfidence[i] = calc_likelihood (frame, xx, yy);
//				
//					cvCircle (frame, cvPoint (xx, yy), 1, CV_RGB (0, 255, 200), -1);
//				}	 
//		}
//		
//		cvCircle(frame,cvPoint(20,20),10,CV_RGB(red,green,blue),-1);
//		cvPutText(frame,"target",cvPoint(0,50),&dfont,CV_RGB(red,green,blue));
//
//		cvShowImage ("Condensation", frame);
// 
//
//		c = cvWaitKey (30);
//    if (c == 27)      break;
//  
//
//		//次のモデルの状態を推定する 
//
//		cvConDensUpdateByTime (cond);
//  } 
//
//	cvDestroyWindow ("Condensation");
//  cvReleaseCapture (&capture);
//
//	cvReleaseImage(&redimage);
//	cvReleaseImage(&greenimage);
//	cvReleaseImage(&blueimage);
//  cvReleaseConDensation (&cond);
//
//
//  cvReleaseMat (&lowerBound);
//  cvReleaseMat (&upperBound);
//  return 0;
//}

//kalman filter
int main (int argc, char **argv){

	cv::VideoCapture capture(0);

	cv::KalmanFilter kalman(4, 2, 0);
	cv::setIdentity(kalman.measurementMatrix, cv::Scalar(1.0));
	cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-5));
	cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(0.1));
	cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1.0));

    // 等速直線運動モデル
	kalman.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
	//観測値を格納するもの
	cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));
	//初期値
	kalman.statePre.at<float>(0) = initx;
	kalman.statePre.at<float>(1) = inity;
	kalman.statePre.at<float>(2) = 0;
	kalman.statePre.at<float>(3) = 0;

	cv::namedWindow("KalmanFilter", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("KalmanFilter", _onMouse, 0);
 
	// メインループ
    while (!GetAsyncKeyState(VK_ESCAPE)) {
        // 画像を取得
		capture >> capframe;
		cv::Mat image = capframe.clone();
 
        // HSVに変換
		//cv::Mat hsv = image.clone();
		//cv::cvtColor(image, hsv, CV_BGR2HSV);


		std::cout << "HSV: (" << h << ", " << s << ", " << v << ")\n";

		//クリックした点のRGBと近い画素を探索し、その重心を求める
		int sumX =0, sumY = 0, counter = 0;

		for(int y = 0; y < image.rows; y++)
			for(int x = 0; x < image.cols; x++)
			{
				//RGBの差分が閾値以内ならカウント
				if(sqrt((red - image.at<cv::Vec3b>(y,x)[2]) * (red - image.at<cv::Vec3b>(y,x)[2]) +
						  (green - image.at<cv::Vec3b>(y,x)[1]) * (green - image.at<cv::Vec3b>(y,x)[1]) +
						  (blue - image.at<cv::Vec3b>(y,x)[0]) * (blue - image.at<cv::Vec3b>(y,x)[0])) <= thresh)
				{
					//std::cout << "    H:" << h << " --- pH:" << ph << std::endl;

					//色付け
					image.at<cv::Vec3b>(y,x)[2] = 255;

					sumX += x;
					sumY += y;
					counter++;
				}
			}
 
		if(counter > 0)
		{
			//予測フェーズ
			cv::Mat prediction = kalman.predict();
			cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

			//観測値
			int mx = (int)(sumX/counter);
			int my = (int)(sumY/counter);
			measurement(0) = mx;
			measurement(1) = my;
			cv::Point measPt(measurement(0), measurement(1));

			//修正フェーズ
			cv::Mat estimated = kalman.correct(measurement);
			cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));

			// 表示
			cv::circle(image, statePt, 10, cv::Scalar(0, 255, 0));
			cv::circle(image, cv::Point(20, 20), 10, cv::Scalar(red, green, blue), 5);
			cv::putText(image,"target", cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(red, green, blue));
        }
  
		std::cout << (int)red << ", " << (int)green << ", " <<  (int)blue << std::endl;
		cv::imshow("KalmanFilter", image);
        cvWaitKey(1);
 
    }
 

	return 0;
}
