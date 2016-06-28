#include "KalmanFilter.h"

void Kalmanfilter::initKalmanfilter()
{
	KF.init(nStates, nMeasurements, nInputs, CV_64F); //init KalmanFilter

	cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
	cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
	cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance

					/* DYNAMIC MODEL */
	//  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
	// position
	KF.transitionMatrix.at<double>(0, 2) = dt;
	KF.transitionMatrix.at<double>(1, 3) = dt;
	KF.transitionMatrix.at<double>(2, 4) = dt;
	KF.transitionMatrix.at<double>(3, 5) = dt;
	KF.transitionMatrix.at<double>(0, 4) = 0.5*pow(dt,2);
	KF.transitionMatrix.at<double>(1, 5) = 0.5*pow(dt,2);

	//KF.transitionMatrix.at<double>(0,3) = dt;
	//KF.transitionMatrix.at<double>(1,4) = dt;
	//KF.transitionMatrix.at<double>(2,5) = dt;
	//KF.transitionMatrix.at<double>(3,6) = dt;
	//KF.transitionMatrix.at<double>(4,7) = dt;
	//KF.transitionMatrix.at<double>(5,8) = dt;
	//KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
	//KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
	//KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
	// orientation
	//KF.transitionMatrix.at<double>(9,12) = dt;
	//KF.transitionMatrix.at<double>(10,13) = dt;
	//KF.transitionMatrix.at<double>(11,14) = dt;
	//KF.transitionMatrix.at<double>(12,15) = dt;
	//KF.transitionMatrix.at<double>(13,16) = dt;
	//KF.transitionMatrix.at<double>(14,17) = dt;
	//KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
	//KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
	//KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
		/* MEASUREMENT MODEL */
	//  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
	//  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
	KF.measurementMatrix.at<double>(0,0) = 1;  // x
	KF.measurementMatrix.at<double>(1,1) = 1;  // y
	//KF.measurementMatrix.at<double>(2,2) = 1;  // z
	//KF.measurementMatrix.at<double>(3,9) = 1;  // roll
	//KF.measurementMatrix.at<double>(4,10) = 1; // pitch
	//KF.measurementMatrix.at<double>(5,11) = 1; // yaw

	//�J���}���t�B���^�̏����l
	KF.statePre.at<double>(0) = 0;
	KF.statePre.at<double>(1) = 0;
}

void Kalmanfilter::fillMeasurements(cv::Mat &measurements, cv::Mat translation_measured)
{
    //// Convert rotation matrix to euler angles
    //cv::Mat measured_eulers(3, 1, CV_64F);
    //measured_eulers = rot2euler(rotation_measured);
    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    //measurements.at<double>(2) = translation_measured.at<double>(2); // z
    //measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    //measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    //measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}

void Kalmanfilter::updateKalmanfilter(cv::Mat measurement, cv::Mat &translation_estimated)
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    //translation_estimated.at<double>(2) = estimated.at<double>(2);
    //// Estimated euler angles
    //cv::Mat eulers_estimated(3, 1, CV_64F);
    //eulers_estimated.at<double>(0) = estimated.at<double>(9);
    //eulers_estimated.at<double>(1) = estimated.at<double>(10);
    //eulers_estimated.at<double>(2) = estimated.at<double>(11);
    //// Convert estimated quaternion to rotation matrix
    //rotation_estimated = euler2rot(eulers_estimated);
}

