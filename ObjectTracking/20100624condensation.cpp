//condensation.cpp
//�N���b�N���������̐F��ǐՂ��܂��B


#include <Windows.h>
#include <stdio.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\legacy\legacy.hpp>
#include <ctype.h>
#include <math.h>
#include <atltime.h>

#include "KalmanFilter.h"

int filterhue;
float sigma =40;

//�N���b�N�����_�̍��W
float initx = 0;
float inity = 0;
//�N���b�N�����_��RGB
unsigned char blue = 0;
unsigned char red = 0;
unsigned char green = 0;
//�N���b�N�������W��HSV
int h = 0;
int s = 0;
int v = 0;
CvScalar targetColor;
IplImage *frame = 0;
IplImage *redimage=0;
IplImage *greenimage=0;
IplImage *blueimage=0;

cv::Mat capframe;
cv::KalmanFilter kalman(4, 2, 0);
//臒l
int thresh = 10;

//�������Ԍv���p
CFileTime cTimeStart, cTimeEnd;
CFileTimeSpan cTimeSpan;

//RGB->HSV�̕ϊ�
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

//�ޓx(�����Ƃ��炵��)�̌v�Z���s�Ȃ��֐�
float calc_likelihood (cv::Mat img, int x, int y)
{
	float dist = 0.0;
    unsigned char b, g, r;
	b = img.at<cv::Vec3b>(y, x)[0];
	g = img.at<cv::Vec3b>(y, x)[1];
	r = img.at<cv::Vec3b>(y, x)[2];
    //b = img->imageData[img->widthStep*y + x*3];     // B
    //g = img->imageData[img->widthStep*y + x*3 + 1]; // G
    //r = img->imageData[img->widthStep*y + x*3 + 2]; // R	
	
	dist = sqrt((float)(blue-b)*(blue-b) +(float)(green-g)*(green-g) +(float)(red-r)*(red-r));
   
	// ����(dist)�𕽋ρAsigma�𕪎U�Ƃ��Ď��A���K���z��ޓx�֐��Ƃ���
	return 1.0 / (sqrt(2.0*CV_PI)*sigma) * expf(-dist*dist/(2.0*sigma*sigma)) ;
}

//particle filter�p
void on_mouse( int event, int x, int y, int flags, void* param )
{
    
    if( event == CV_EVENT_LBUTTONDOWN )
    {
		  //if(frame->nChannels==3)
		if(capframe.channels() == 3)
		{
		 		int r,g,b,count;
				r=g=b=0;
				count=0;
				for(int i=-3;i<3;i++)
				{			
					for(int j=-3;j<3;j++)
					{
						b += capframe.at<cv::Vec3b>(y+j, x+i)[0];
						g += capframe.at<cv::Vec3b>(y+j, x+i)[1];
						r += capframe.at<cv::Vec3b>(y+j, x+i)[2];
						//b =b+ frame->imageData[frame->widthStep*(y+j) +(x+i)*3];     // B
						//g =g+ frame->imageData[frame->widthStep*(y+j) +(x+i)*3 + 1]; // G
						//r =r+ frame->imageData[frame->widthStep*(y+j) +(x+i)*3 + 2]; // R
						count++;
					}
				}
				
				//���ϒl�������Ώۂɂ���
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
		//�J���}���t�B���^�̏����l
		//kalman.statePre.at<float>(0) = initx;
		//kalman.statePre.at<float>(1) = inity;

		//�ŏ��Ɉ��\��
		//cv::Mat prediction = kalman.predict();

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
				
				//���ϒl�������Ώۂɂ���
				blue = sumB/count;
				green = sumG/count;
				red = sumR/count;
    }
}

//�p�[�e�B�N���t�B���^
void particleFilter()
{
	int i, c;
	double w = 0.0, h = 0.0;
	cv::VideoCapture capture(0);
	//CvCapture *capture = 0;
	//capture = cvCreateCameraCapture (0);

	  int n_stat = 4;
	  int n_particle = 4000;
	  CvConDensation *cond = 0;
	  CvMat *lowerBound = 0;
	  CvMat *upperBound = 0;
	  int xx, yy;

	  capture >> capframe;

	//�P�t���[���L���v�`�����C�L���v�`���T�C�Y���擾����D
	//frame = cvQueryFrame (capture);
	//redimage=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	//greenimage=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	//blueimage=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
	w = capframe.cols;
	h = capframe.rows;
	//w = frame->width;
    //h = frame->height;
	cv::namedWindow("Condensation", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("Condensation", on_mouse, 0);
	//cvNamedWindow ("Condensation", CV_WINDOW_AUTOSIZE);
	//cvSetMouseCallback("Condensation",on_mouse,0);
 
 	//�t�H���g�̐ݒ�
	//CvFont dfont;
	//float hscale      = 0.7f;
	//float vscale      = 0.7f;
	//float italicscale = 0.0f;
	//int  thickness    = 1;
	//char text[255] = "";
	//cvInitFont (&dfont, CV_FONT_HERSHEY_SIMPLEX , hscale, vscale, italicscale, thickness, CV_AA); 

	//Condensation�\���̂��쐬����D
	cond = cvCreateConDensation (n_stat, 0, n_particle);

	//��ԃx�N�g���e�����̎�肤��ŏ��l�E�ő�l���w�肷��
	//����͈ʒu(x,y)�Ƒ��x(xpixcel/frame,ypixcel/frame)��4����
	lowerBound = cvCreateMat (4, 1, CV_32FC1);
	upperBound = cvCreateMat (4, 1, CV_32FC1);
	cvmSet (lowerBound, 0, 0, 0.0);
	cvmSet (lowerBound, 1, 0, 0.0);
	cvmSet (lowerBound, 2, 0, -20.0);
	cvmSet (lowerBound, 3, 0, -20.0);
	cvmSet (upperBound, 0, 0, w);
	cvmSet (upperBound, 1, 0, h);
	cvmSet (upperBound, 2, 0, 20.0);
	cvmSet (upperBound, 3, 0, 20.0);
  
	//Condensation�\���̂�����������
	cvConDensInitSampleSet (cond, lowerBound, upperBound);

	//ConDensation�A���S���Y���ɂ������ԃx�N�g���̃_�C�i�~�N�X���w�肷��
	cond->DynamMatr[0] = 1.0;
	cond->DynamMatr[1] = 0.0;
	cond->DynamMatr[2] = 1.0;
	cond->DynamMatr[3] = 0.0;
	cond->DynamMatr[4] = 0.0;
	cond->DynamMatr[5] = 1.0;
	cond->DynamMatr[6] = 0.0;
	cond->DynamMatr[7] = 1.0;
	cond->DynamMatr[8] = 0.0;
	cond->DynamMatr[9] = 0.0;
	cond->DynamMatr[10] = 1.0;
	cond->DynamMatr[11] = 0.0;
	cond->DynamMatr[12] = 0.0;
	cond->DynamMatr[13] = 0.0;
	cond->DynamMatr[14] = 0.0;
	cond->DynamMatr[15] = 1.0;
  
	//�m�C�Y�p�����[�^���Đݒ肷��D
	cvRandInit (&(cond->RandS[0]), -25, 25, (int) cvGetTickCount ());
	cvRandInit (&(cond->RandS[1]), -25, 25, (int) cvGetTickCount ());
	cvRandInit (&(cond->RandS[2]), -5, 5, (int) cvGetTickCount ());
	cvRandInit (&(cond->RandS[3]), -5, 5, (int) cvGetTickCount ());
	
	while (1) 
	{
		capture >> capframe;
		//frame = cvQueryFrame (capture);


		//�e�p�[�e�B�N���ɂ��Ėޓx���v�Z����D
		for (i = 0; i < n_particle; i++)
		{ 
			xx = (int) (cond->flSamples[i][0]);
			yy = (int) (cond->flSamples[i][1]);
			 if (xx < 0 || xx >= w || yy < 0 || yy >= h) 
				{  
					cond->flConfidence[i] = 0.0;
				}
				else
				{  
					cond->flConfidence[i] = calc_likelihood (capframe, xx, yy);
					//cond->flConfidence[i] = calc_likelihood (frame, xx, yy);
				
					cv::circle(capframe, cv::Point(xx, yy), 1, CV_RGB(0, 255, 200));
					//cvCircle (frame, cvPoint (xx, yy), 1, CV_RGB (0, 255, 200), -1);
				}	 
		}

		//�d�݂̑��a&�d�S�����߂�
		double wx = 0, wy = 0;
		double sumWeight = 0;
		for (i = 0; i < n_particle; i++)
		{
			sumWeight += cond->flConfidence[i];
		}
		for (i = 0; i < n_particle; i++)
		{
			wx += (int) (cond->flSamples[i][0]) * (cond->flConfidence[i] / sumWeight);
			wy += (int) (cond->flSamples[i][1]) * (cond->flConfidence[i] / sumWeight);
		}

		//�d�S�\��
		cv::circle(capframe, cv::Point((int)wx, (int)wy), 10, cv::Scalar(0, 0, 255));
		cv::circle(capframe, cv::Point(20, 20), 10, CV_RGB(red, green, blue), 6);
		cv::putText(capframe, "target", cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(red, green, blue));
		cv::imshow("Condensation", capframe);

		//cvCircle(frame,cvPoint(20,20),10,CV_RGB(red,green,blue),-1);
		//cvPutText(frame,"target",cvPoint(0,50),&dfont,CV_RGB(red,green,blue));
		//cvShowImage ("Condensation", frame);
		
		c = cv::waitKey(30);
		//c = cvWaitKey (30);
		if (c == 27)      break;
  
		//���̃��f���̏�Ԃ𐄒肷�� 
		cvConDensUpdateByTime (cond);

	}

	cv::destroyWindow("Condensation");
	//cvDestroyWindow ("Condensation");
	//cvReleaseCapture (&capture);

	//cvReleaseImage(&redimage);
	//cvReleaseImage(&greenimage);
	//cvReleaseImage(&blueimage);
	cvReleaseConDensation (&cond);

	cvReleaseMat (&lowerBound);
	cvReleaseMat (&upperBound);
}

//�J���}���t�B���^
void kalmanFilter()
{
	cv::VideoCapture capture(0);

	//x,y,vx,vy
	//cv::KalmanFilter kalman(4, 2, 0);
	cv::setIdentity(kalman.measurementMatrix, cv::Scalar(1.0));
	cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-5));
	cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(0.1));
	cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1.0));

    // ���������^�����f��
	kalman.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
	//�ϑ��l���i�[�������
	cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));
	//�����l
	kalman.statePre.at<float>(0) = initx;
	kalman.statePre.at<float>(1) = inity;
	kalman.statePre.at<float>(2) = 0;
	kalman.statePre.at<float>(3) = 0;

	cv::namedWindow("KalmanFilter", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("KalmanFilter", _onMouse, 0);

	// ���C�����[�v
    while (!GetAsyncKeyState(VK_ESCAPE)) {
        // �摜���擾
		capture >> capframe;
		cv::Mat image = capframe.clone();
 
        // HSV�ɕϊ�
		//cv::Mat hsv = image.clone();
		//cv::cvtColor(image, hsv, CV_BGR2HSV);

		std::cout << "HSV: (" << h << ", " << s << ", " << v << ")\n";

		//�N���b�N�����_��RGB�Ƌ߂���f��T�����A���̏d�S�����߂�
		int sumX =0, sumY = 0, counter = 0;

		for(int y = 0; y < image.rows; y++)
			for(int x = 0; x < image.cols; x++)
			{
				//RGB�̍�����臒l�ȓ��Ȃ�J�E���g
				if(sqrt((red - image.at<cv::Vec3b>(y,x)[2]) * (red - image.at<cv::Vec3b>(y,x)[2]) +
						  (green - image.at<cv::Vec3b>(y,x)[1]) * (green - image.at<cv::Vec3b>(y,x)[1]) +
						  (blue - image.at<cv::Vec3b>(y,x)[0]) * (blue - image.at<cv::Vec3b>(y,x)[0])) <= thresh)
				{
					//�F�t��
					image.at<cv::Vec3b>(y,x)[2] = 255;

					sumX += x;
					sumY += y;
					counter++;
				}
			}
 
		if(counter > 0)
		{
			//�ϑ��l
			int mx = (int)(sumX/counter);
			int my = (int)(sumY/counter);
			measurement(0) = mx;
			measurement(1) = my;
			cv::Point measPt(measurement(0), measurement(1));
			//�\��
			cv::circle(image, measPt, 10, cv::Scalar(0, 0, 255));

			//�C���t�F�[�Y
			cv::Mat estimated = kalman.correct(measurement);
			cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));

        }

		//�\���t�F�[�Y
		cv::Mat prediction = kalman.predict();
		cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
		// �\��
		cv::circle(image, predictPt, 10, cv::Scalar(0, 255, 0));
		cv::circle(image, cv::Point(20, 20), 10, CV_RGB(red, green, blue), 5);
		cv::putText(image,"target", cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(red, green, blue));
  
		std::cout << (int)red << ", " << (int)green << ", " <<  (int)blue << std::endl;
		cv::imshow("KalmanFilter", image);
        cvWaitKey(1);
 
    }
}

//�J���}���t�B���^�N���X�e�X�g
void KFtest()
{
	cv::VideoCapture capture(0);

	cv::namedWindow("KalmanFilter", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("KalmanFilter", _onMouse, 0);

	Kalmanfilter kf(6, 2, 0, 0.03);
	kf.initKalmanfilter();

	// ���C�����[�v
    while (!GetAsyncKeyState(VK_ESCAPE)) {

		//�������Ԍv���J�n
		cTimeStart = CFileTime::GetCurrentTime();// ���ݎ���

        // �摜���擾
		capture >> capframe;
		cv::Mat image = capframe.clone();
 

		//std::cout << "HSV: (" << h << ", " << s << ", " << v << ")\n";

		//�N���b�N�����_��RGB�Ƌ߂���f��T�����A���̏d�S�����߂�
		int sumX =0, sumY = 0, counter = 0;

		for(int y = 0; y < image.rows; y++)
			for(int x = 0; x < image.cols; x++)
			{
				//RGB�̍�����臒l�ȓ��Ȃ�J�E���g
				if(sqrt((red - image.at<cv::Vec3b>(y,x)[2]) * (red - image.at<cv::Vec3b>(y,x)[2]) +
						  (green - image.at<cv::Vec3b>(y,x)[1]) * (green - image.at<cv::Vec3b>(y,x)[1]) +
						  (blue - image.at<cv::Vec3b>(y,x)[0]) * (blue - image.at<cv::Vec3b>(y,x)[0])) <= thresh)
				{
					//�F�t��
					image.at<cv::Vec3b>(y,x)[2] = 255;

					sumX += x;
					sumY += y;
					counter++;
				}
			}
 
		if(counter > 0)
		{
			//�ϑ��l
			int mx = (int)(sumX/counter);
			int my = (int)(sumY/counter);
			cv::Mat translation_measured(2, 1, CV_64F);
			translation_measured.at<double>(0, 0) = mx;
			translation_measured.at<double>(1, 0) = my;
			
			cv::Mat measurement(2, 1, CV_64F);
			kf.fillMeasurements(measurement, translation_measured);

			// Instantiate estimated translation and rotation
			cv::Mat translation_estimated(2, 1, CV_64F);
			// update the Kalman filter with good measurements
			kf.updateKalmanfilter(measurement, translation_estimated);

			//�\��
			cv::circle(image, cv::Point(mx, my), 10, cv::Scalar(0, 0, 255));
			cv::circle(image, cv::Point(translation_estimated.at<double>(0, 0), translation_estimated.at<double>(1,0)), 10, cv::Scalar(0, 255, 0));
        }


		cv::putText(image,"target", cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(red, green, blue));
  
		//std::cout << (int)red << ", " << (int)green << ", " <<  (int)blue << std::endl;
		cv::imshow("KalmanFilter", image);
        cvWaitKey(1);

		cTimeEnd = CFileTime::GetCurrentTime();
		cTimeSpan = cTimeEnd - cTimeStart;
		std::cout<< "1frame��������:" << cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;
    }
}

int main (int argc, char **argv){

	//kalmanFilter();
	//particleFilter();

	KFtest();

	return 0;
}
