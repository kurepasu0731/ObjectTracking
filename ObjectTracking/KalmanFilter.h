#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#pragma once

#include <opencv2\opencv.hpp>

class Kalmanfilter
{
public:
	cv::KalmanFilter KF;
	int nStates; //��ԃx�N�g���̎�����
	int nMeasurements; //�ϑ��x�N�g���̎�����
	int nInputs; //the number of action control
	double dt; //time between measurements(1/FPS)

	Kalmanfilter(int _nStates, int _nMeasurements, int _nInputs, int _dt)
		: nStates(_nStates)
		, nMeasurements(_nMeasurements)
		, nInputs(_nInputs)
		, dt(_dt)
	{
	}

	~Kalmanfilter(){};

	//������
	void initKalmanfilter();

	//�ϑ��l�̐��`
	void fillMeasurements(cv::Mat &measurements, cv::Mat translation_measured);

	//�ϑ��l��o�^���A�X�V�A�\���l�̎擾
	void updateKalmanfilter(cv::Mat measurement, cv::Mat &translation_estimated);

};

#endif