#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <windows.h>
#include <limits>

using namespace std;
using namespace cv;

ofstream csvFile;

// Matrices
Mat src, gray, blobs, segmented;
int cols = 640;
int rows = 480;
Mat trajectory_mat = Mat::zeros(rows, cols, CV_8UC3);
Mat predictedTrajectory_mat = Mat::zeros(rows, cols, CV_8UC3);

// classification and logging
String previousClassification, classification;
string video_filename, take, image_filename;
string path = "./";

int patient;

// Blob detection & tracking
vector<Point> acceptedBlobs;
vector<Point> trajectory;
vector<Point> velocity;
Point prediction;

int frame = 0;
int frameRate = 30;

bool showOperations, showErosion, showClosing, showThresholding, useGT;

// ROI
int horizontal = 190;
int vertical = rows / 2 + 30;

// ground truth
int trans1 = 0;
int trans2 = 0;
int standingGT = 0;
int sittingGT = 0;
string frameLabel;

// spatial treshold for the different participants
int sitting[4] = { 210, 228,218,189 };
int standing[4] = { 175,207,174,189 };

// Evaluation
int correctClassifications = 0;
float recall, precision, classifications, evaluatedFrames, accuracy, trueNegativeRate, fScore;
float truePositives, trueNegatives, falsePositives, falseNegatives;
String groundTruth;

// for quality assessment:
double distanceThreshold = 100;
double distThres = distanceThreshold;
bool accepted;

const int threshold_slider_max = 255;
int threshold_slider = 211;

void on_trackbar_threshold(int, void*) {
	threshold(src, segmented, threshold_slider, 255, CV_THRESH_BINARY);
	imshow("Thresholded", segmented);
}

const int erosion_slider_max = 10;
int erosion_slider = 0;

void on_trackbar_erode(int, void*) {
	int erosion = 1 + erosion_slider * 2;
	Mat erosionElement = getStructuringElement(MORPH_ELLIPSE, Size(erosion, erosion), Point(-1, -1));
	erode(segmented, segmented, erosionElement);
	imshow("Eroded", segmented);
}

const int closing_slider_max = 10;
int closing_slider = 2;

void on_trackbar_close(int, void*) {
	int close = 1 + closing_slider * 2;
	Mat closingElement = getStructuringElement(MORPH_ELLIPSE, Size(close, close), Point(-1, -1));
	morphologyEx(segmented, segmented, 2, closingElement);
	imshow("Closed", segmented);
}

double distance(Point point1, Point point2) {
	Point diff = point2 - point1;
	return sqrt(abs(diff.x * diff.x + diff.y * diff.y));
}

void log(int currentFrame, string classification, string groundTruth, bool accepted, string video) {
	csvFile.open("data.csv", ios::app);
	if (csvFile.is_open()) {
		csvFile << currentFrame << ";"
			<< classification << ";"
			<< groundTruth << ";"
			<< accepted << ";"
			<< video << "\n";
	}
	else {
		cout << "Could not write to data.\n";
	}
	csvFile.close();
}

Point calculatePrediction(vector<Point> trajectory) {
	Point temp;
	if (trajectory.size() > 1) {
		Point vel = trajectory.at(trajectory.size() - 2) - trajectory.back();
		velocity.push_back(vel);
		if (velocity.size() > 1) {
			Point acceleration = velocity.at(velocity.size() - 2) - velocity.back();
			temp.x = 0.5 * acceleration.x * velocity.back().x + trajectory.back().x;
			temp.y = 0.5 * acceleration.y * velocity.back().y + trajectory.back().y;
		}
	}
	return temp;
}

void saveImage(Mat img, int currentFrame, string classification) {
	string path = "./images/";
	image_filename = path;
	image_filename.append(video_filename, 0, 20);
	image_filename.append("frame");
	image_filename.append(to_string(frame));
	image_filename.append(classification);
	imwrite("./" + image_filename + ".jpg", src);
}

void drawTrajectory(Mat src, int currentFrame, Point p) {
	Scalar colour;
	Scalar transientColour(125, 125, 125);
	Scalar sittingColour(255, 0, 0);
	Scalar standingColour(0, 255, 0);
	Scalar otherColour(0, 0, 255);

	if (currentFrame > sittingGT) {
		colour = sittingColour;
	}	else if (currentFrame > trans2) {
		colour = transientColour;
	}
	else if (currentFrame > standingGT) {
		colour = standingColour;
	}
	else if (currentFrame > trans1) {
		colour = transientColour;
	}
	else {
		colour = otherColour;
	}
	// assign pixel to colour 
	//src.at<Vec3b>(p) = colour;
}

int main(int argc, const char** argv)
{
	/*********************/
	/* INITIALISE VALUES */
	/*********************/
	video_filename = "camera1patient";
	cout << "Choose recording" << endl << "Patient (1-4): ";
	cin >> patient;
	video_filename.append(to_string(patient));
	video_filename.append("take");
	cout << "Take (1-3): ";
	cin >> take;
	video_filename.append(take);
	video_filename.append(".wmv");
	cout << "Video: " << video_filename << endl << "*******************************" << endl;

	VideoCapture cap(video_filename);
	if (!cap.isOpened()) return -1;

	cout << "Show operations? ";
	cin >> showOperations;

	if (showOperations) {
		cout << "Show Thresholding? ";
		for (;;)
		{
			if (cin >> showThresholding) break;
			cout << "Valid input: 0 or 1. Try again ";
			cin.clear();
		}

		cout << "Show Erosion? ";
		for (;;)
		{
			if (cin >> showErosion) break;
			cout << "Valid input: 0 or 1. Try again ";
			cin.clear();
		}

		cout << "Show Closing? ";
		for (;;)
		{
			if (cin >> showClosing) break;
			cout << "Valid input: 0 or 1. Try again ";
			cin >> showClosing;
			cin.clear();
		}
	}
	cout << "Ground Truth? ";
	for (;;)
	{
		if (cin >> useGT) break;
		cout << "Valid input: 0 or 1. Try again ";
		cin.clear();
	}

	if (useGT) {
		cout << "> Ground Truth: Other/Standing transient frame: ";
		cin >> trans1;
		cout << "> Ground Truth: Standing frame: ";
		cin >> standingGT;
		cout << "> Ground Truth: Standing/Sitting transient frame: ";
		cin >> trans2;
		cout << "> Ground Truth: Sitting frame: ";
		cin >> sittingGT;
		cin.clear();
	}


	cout << "*******************************" << endl;

	for (;;)
	{
		int yLast = INT_MAX;
		double smallestDist = DBL_MAX;
		int videoTime = frame / frameRate;
		cap >> src;
		if (src.empty()) break;

		if (useGT) {
			if (frame >= sittingGT) {
				groundTruth = "Sitting";
			}
			else if (frame >= trans2) {
				groundTruth = "Transient";
			}
			else if (frame >= standingGT) {
				groundTruth = "Standing";
			}
			else if (frame >= trans1) {
				groundTruth = "Transient";
			}
			else {
				groundTruth = "Other";
			}
		}

		/****************/
		/* SEGMENTATION */
		/****************/
		// Threshold
		if (showThresholding) {
			namedWindow("Thresholded", 1);
			createTrackbar("Threshold", "Thresholded", &threshold_slider, threshold_slider_max, on_trackbar_threshold);
			on_trackbar_threshold(threshold_slider, 0);
		}
		else {
			threshold(src, segmented, threshold_slider, 255, CV_THRESH_BINARY);
		}

		// Erosion
		if (showErosion) {
			namedWindow("Eroded", 1);
			createTrackbar("Erosion", "Eroded", &erosion_slider, erosion_slider_max, on_trackbar_erode);
			on_trackbar_erode(erosion_slider, 0);
		}
		else {
			Mat erosionElement = getStructuringElement(MORPH_ELLIPSE, Size(erosion_slider * 2 + 1, erosion_slider * 2 + 1), Point(-1, -1));
			erode(segmented, segmented, erosionElement);
		}

		// Closing
		if (showClosing) {
			namedWindow("Closed", 1);
			createTrackbar("Closing", "Closed", &closing_slider, closing_slider_max, on_trackbar_close);
			on_trackbar_close(closing_slider, 0);
		}
		else {
			Mat closingElement = getStructuringElement(MORPH_ELLIPSE, Size(closing_slider * 2 + 1, closing_slider * 2 + 1), Point(-1, -1));
			morphologyEx(segmented, segmented, 2, closingElement);
		}

		/****************/
		/* BLOBANALYSIS */
		/****************/
		SimpleBlobDetector::Params params;

		// Blob colour
		params.blobColor = 255;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = 20;
		params.maxArea = 500000;

		// Filter by Circularity
		params.filterByCircularity = false;
		params.minCircularity = 0.01f;
		//maxCircularity = 1;

		// Filter by Convexity
		params.filterByConvexity = false;
		params.minConvexity = 0.1f;
		//params.maxConvexity = 1;

		// Filter by Inertia
		params.filterByInertia = false;
		params.minInertiaRatio = 0.01f;

		// Detect blobs
		Ptr<SimpleBlobDetector> sbd = SimpleBlobDetector::create(params);
		vector<KeyPoint> keypoints;
		sbd->detect(segmented, keypoints);
		drawKeypoints(src, keypoints, blobs, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		/******************/
		/* CLASSIFICATION */
		/******************/
		if (keypoints.size() > 0) {
			KeyPoint patientHead;

			// Prediction from last frame.
			if (prediction.x != NULL) {
				Rect ROIPrediction(prediction.x - 50, prediction.y - 50, 100, 100);
				for (int i = 0; i < keypoints.size(); i++) {
					if (ROIPrediction.contains(keypoints[i].pt))
						if (distance(keypoints[i].pt, prediction) < smallestDist) {
							smallestDist = distance(keypoints[i].pt, prediction);
							prediction = keypoints[i].pt;
						}
				}
				Rect ROIPrediction2(prediction.x - 50, prediction.y - 50, 100, 100);
				rectangle(blobs, ROIPrediction2, Scalar(0, 255, 0), 1, 8, 0);
			}

			// Loop through all blobs in the vector and find the one with the least y = the head of the patient.
			for (int i = 0; i < keypoints.size(); i++) {
				int yNew = keypoints[i].pt.y;
				if (yLast > yNew) {
					yLast = yNew;
					patientHead = keypoints[i];
				}
			}

			// Check where the blob is.
			horizontal = (sitting[patient - 1] + standing[patient - 1]) / 2;
			line(blobs, Point(vertical, horizontal), Point(cols, horizontal), Scalar(255, 0, 0), 1);
			line(blobs, Point(vertical, 0), Point(vertical, rows), Scalar(255, 0, 0), 1);

			// Classify the frame.
			if (patientHead.pt.x < vertical) {
				classification = "Other";
			}
			else if (patientHead.pt.y < horizontal) {
				classification = "Standing";
			}
			else {
				if(previousClassification != "Other") 
				classification = "Sitting";
			}
			
			bool newClassification = classification != previousClassification ? true : false;

			if (newClassification) {
				saveImage(src, frame, classification);
				previousClassification = classification;
			}

			/************/
			/* TRACKING */
			/************/
			trajectory.push_back(patientHead.pt);

			if (trajectory.size() > 1) {
				Point vel = trajectory.at(trajectory.size() - 2) - trajectory.back();
				velocity.push_back(vel);
				if (velocity.size() > 1) {
					Point acceleration = velocity.at(velocity.size() - 2) - velocity.back();
					prediction.x = 0.5 * acceleration.x * velocity.back().x + trajectory.back().x;
					prediction.y = 0.5 * acceleration.y * velocity.back().y + trajectory.back().y;
				}
			}
			prediction = calculatePrediction(trajectory);

			/**********************/
			/* QUALITY ASSESSMENT */
			/**********************/
			Point currentBlob = trajectory.back();

			// if it is the first blob detected, set it as the reference point.
			if (acceptedBlobs.size() == 0) {
				acceptedBlobs.push_back(currentBlob);
			}

			// calculate the distance from the last accepted blob to the current blob.
			double dist = distance(acceptedBlobs.back(), currentBlob);
			if (dist < distanceThreshold) {
				if (groundTruth != "Transient") {
					classifications++;
					if (classification == groundTruth) {
						correctClassifications++;
					}
				}				
				acceptedBlobs.push_back(currentBlob);
				accepted = true;
				trajectory_mat.at<Vec3b>(patientHead.pt) = 255;
				distanceThreshold = distThres;
			}
			else {
				accepted = false;
				distanceThreshold += dist;	// abs(prediction)
			}
				
			// Blue box around the position of the last accepted blob
			Rect lastAccepted(acceptedBlobs.back().x - 50, acceptedBlobs.back().y - 50, 100, 100);
			rectangle(blobs, lastAccepted, Scalar(255, 0, 0), 1, 8, 0);

			// Green box around the position of the topmost blob
			Rect topBlob(patientHead.pt.x - 50, patientHead.pt.y - 50, 100, 100);
			rectangle(blobs, topBlob, Scalar(0, 0, 255), 1, 8, 0);
		}
		imshow("Blobs", blobs);
		if(useGT) log(frame, classification, groundTruth, accepted, video_filename);
			
		frame++;
		if (groundTruth != "Transient") evaluatedFrames++;

		char k = (char)waitKey(30);
		if (k == 27) break;
	}
	if (correctClassifications != 0 && useGT) {
		truePositives = correctClassifications;
		falsePositives = evaluatedFrames - correctClassifications;
		trueNegatives = 0;
		falseNegatives = evaluatedFrames - classifications;
		precision = correctClassifications / classifications;
		recall = correctClassifications / evaluatedFrames;
		accuracy = truePositives / (truePositives + falsePositives + falseNegatives);
		trueNegativeRate = trueNegatives / (trueNegatives + falseNegatives);
		fScore = 2 * ((precision*recall) / (precision + recall));
	}
	csvFile.open("data.csv", ios::app);
	if (csvFile.is_open()) {
		csvFile << "Precision: " << precision << "\n"
			<< "Recall: " << recall << "\n"
			<< "Accuracy: " << accuracy << "\n"
			<< "F score: " << fScore << "\n";
	}
	else {
		cout << "Could not write to data.\n";
	}
	csvFile.close();
	return 0;
}
