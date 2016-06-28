#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#pragma once

#include <opencv2\opencv.hpp>

class Kalmanfilter
{
public:
	cv::KalmanFilter KF;
	int nStates; //状態ベクトルの次元数
	int nMeasurements; //観測ベクトルの次元数
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

	//初期化
	void initKalmanfilter();

	//観測値の成形
	void fillMeasurements(cv::Mat &measurements, cv::Mat translation_measured);

	//観測値を登録し、更新、予測値の取得
	void updateKalmanfilter(cv::Mat measurement, cv::Mat &translation_estimated);

};

#endif