#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"				// rects, used for the final displayed image with rectangles.
#include <math.h>											// math functions, rand().
#include <iostream>											// cin and cout.											
#include <stdio.h>											// variable size_t.
#include <stdlib.h>											// also size_t.

using namespace std;
using namespace cv;

// CLASSES //
class Dice{
private:
	bool roll = false;
	int diceRoll = 0;
	int whiteThreshold = 500;
public:
	int rollDice(){
		// calculates a random number 1-6 and returns it.
		diceRoll = rand() % 6 + 1;
		return diceRoll;
	}
	bool isRolled(){
		// if dice was not rolled, dice roll is set to zero. Return isRolled boolean.
		if (!roll) diceRoll = 0;
		return roll;
	}
	int getDiceRoll(){
		// returns the roll of the dice.
		return diceRoll;
	}
	int update(Mat mat){
		bool check;
		int whitePixels = 0;

		// goes through pixels in the passed image.
		for (size_t y = 0; y < mat.rows; y++) {
			for (size_t x = 0; x < mat.cols; x++){
				uchar* ptr = (uchar*)(mat.data);
				uchar val = ptr[mat.step * y + x];
				// if a pixel is white increment whitePixels.
				if (val == 255)	whitePixels++;
			}
		}
		// if the number of white pixels is above the white threshold, make a dice roll.
		if (whitePixels < whiteThreshold) roll = false;
		else roll = true;

		if (roll) diceRoll = rollDice();
		else diceRoll = 0;

		// same as:
		/*
		if(whitePixels < whiteThreshold){
			roll = false;
			diceRoll = 0;
		}
		else diceRoll = rollDice();
		*/

		return diceRoll;
	}
} dice;
class Piece{
private:
	int tile;									// the given tile that the piece is on
	int prevTile;									// last tile the piece was on, used to call tilesMoved.
	int tilesMoved = 0;								// last number of tiles moved
	int projectingTile;								// the tile of the piece + dice number
	bool detected = false;
	Point2i pos;									// position of piece.
public:
	int getTile(){
		return tile;
	}
	int getProjectionTile(int dice){
		// checks if the dice roll + tile is beyound the bounderies, if it is, - 15.
		// e.g. roll = 6, tile = 15, projection tile = 6.
		if (tile + dice > 15){
			projectingTile = tile + dice - 15;
		}
		// if it is inside of the bounderies it just becomes the dice + the tile.
		else{
			projectingTile = tile + dice;
			return projectingTile;
		}
	}
	bool wasDetected(){
		// returns the boolean detected.
		return detected;
	}
	
	void updatePos(vector<Point2i> tiles, vector<Point2i> point){			
		// finds the shortest euclidean distance between tiles and the passed point.
		// in other words, it finds the closest tile to the piece.

		pos.x = point.at(0).x;											
		pos.y = point.at(0).y;
		// if the point is (0,0) detection is false.
		if (pos.x == 0 && pos.y && 0) detected = false;
		else detected = true;

		int distance = 99999;
		// start by setting distance as an extremely high number, 
		// as a new distance will be set uf temp is less than the previous distance.

		prevTile = tile;								// start by setting the position to old

		// check if there is enough tiles.
		if (tiles.size() == 15){
			// from 0 - 14
			for (int i = 0; i < tiles.size(); i++){					// for all points in the vector
				int x = pos.x - tiles.at(i).x;					// needs to be positive always
				int y = pos.y - tiles.at(i).y;					// ...or compare temp to be closer to zero
				int temp = sqrt(pow(x, 2) + pow(y, 2));				// will be compared to the previous shortest distance

				// distance is only updated if it is less than the previous distance.
				// as we are only interested in the shortest distance,
				if (temp < distance) {
					distance = temp;						// temp is set as the distance if it is the shortest.
					tile = i;							// the tile of the shortest distance is the tile of the piece.
				}
			}
			// is actually not used.
			tilesMoved = tile - prevTile;
		}
		// this will happen if there isn't enough tiles.
		else{
			cout << "Not enough tiles detected to update piece position." << endl;
			cout << endl;
			tile = prevTile;
		}
	}
};
class Segmentation{
private:
	Mat segmentMatrix;								// the matrix to be segmented.
	Mat hsv;									// the matrix for the converted hsv coolourspace
	Mat dst = Mat::zeros(480, 640, 0);						// the destination matrix.
	Mat empty = Mat::zeros(480, 640, 0);						// just an empty matrix for erosion.
	int width;																		// width of matrix.
	int height;																		// height of matrix.
	int thres1;																		// threshold value 1.
	int thres2;																		// threshold value 2.
	int iterations;																	// iterations for erosion and dilation.
	int medianFilterIterations;														// iterations for median filter.
public:
	Segmentation(int int1, int int2, int int3, int int4){						// constructor
		thres1 = int1;
		thres2 = int2;
		iterations = int3;
		medianFilterIterations = int4;
	}
	Mat segmentHue(Mat mat){
		segmentMatrix = mat.clone();						// segment matrix becomes a clone of the passed matrix
		hsv = mat.clone();							// hsv becomes as clone of the passed matrix
		width = segmentMatrix.cols;						// set width
		height = segmentMatrix.rows;						// set height
		thresholdHue();								// threshold hue
		erosion();								// erode
		dilation();								// dilate	
		medianFilter();								// median filter
		return segmentMatrix;							// return the segmented matrix
	}
	Mat segmentValue(Mat mat){
		segmentMatrix = mat.clone();						// segment matrix becomes a clone of the passed matrix
		hsv = mat.clone();							// hsv becomes as clone of the passed matrix
		width = segmentMatrix.cols;						// set width
		height = segmentMatrix.rows;						// set height
		thresholdValue();							// threshold on value
		erosion();								// erode
		dilation();								// dilate
		medianFilter();								// median filter
		return segmentMatrix;							// return segmented image
	}
protected:										//can be 'protected' - no global access, but not as private as private
	void thresholdHue() {
		for (size_t y = 0; y < height; y++){					// for every pixel in the image
			for (size_t x = 0; x < width; x++){
				int valueInput = hsv.at<Vec3b>(y, x)[0];				// input value becomes the hue channel of that pixel.
				if (valueInput < thres2 && valueInput >= thres1)			// if the value of the input is less than the upper threshold and greater than the lower threshold...
					dst.at<uchar>(y, x) = 255;					// set it to white...
				else
					dst.at<uchar>(y, x) = 0;					// otherwise set it to black.
			}
		}
		segmentMatrix = dst.clone();								// update the segmented matrix.
	}
	void thresholdValue() {																			
		for (size_t y = 0; y < height; y++){							// for every pixel in the image.
			for (size_t x = 0; x < width; x++){
				int valueInput = hsv.at<Vec3b>(y, x)[2];				// input value becomes the value of the value channel.
				if (valueInput < thres2 && valueInput >= thres1)			// threshold.
					dst.at<uchar>(y, x) = 255;
				else
					dst.at<uchar>(y, x) = 0;
			}
		}
		segmentMatrix = dst.clone();								// update the segmented matrix
	}
	void erosion(){
		Mat temp = empty.clone();													
		int counter = 0;

		while (counter < iterations){								// while iterations are not met...
			for (int y = 1; y < height; y++){						// for each pixel in the image...
				for (int x = 1; x < width; x++){

					int valueIn = getPointer(x, y);
					if (valueIn == 255); {						// if the pixel is white...
						for (uchar i = 0; i < 9; i++){				// check all the neighbours
							int tempY = i / 3 - 1 + y;
							int tempX = i % 3 - 1 + x;
							uchar valueOut = getPointer(tempX, tempY);
							if (valueOut == 0) break;					// it a neighbour is black, break
							if (i == 8) temp.at<uchar>(y, x) = 255;				// if all the neighbours are white, set the original pixel in the black matrix to be white
						}
					}
				}
			}
			segmentMatrix = temp.clone();									// segmented matrix becomes the checked matrix
			temp = empty.clone();										// temp becomes empty
			counter++;											// increment how many times erosion has been applied
		}
	}
	void dilation(){
		Mat temp = segmentMatrix.clone();
		int counter = 0;

		while (counter < iterations){										// while iterations are not met...
			for (int y = 1; y < height; y++){								// for every pixel...
				for (int x = 1; x < width; x++){
					uchar valueIn = getPointer(x, y);								
					if (valueIn == 255){										// if the pixel is white...
						for (uchar i = 0; i < 9; i++){								// for all the neighbours
							int tempY = i / 3 - 1 + y;
							int tempX = i % 3 - 1 + x;
							if (tempX < width && tempY < height) temp.at<uchar>(tempY, tempX) = 255;	// set neighbours to be white in temp...
						}
					}
				}
			}
			segmentMatrix = temp.clone();							// set segmented matrix to be a temp clone
			counter++;									// increment how many times erosion has been applied
		}
	}
	void medianFilter(){
		Mat temp = empty.clone();
		int counter = 0;
		bool continuousCheck = segmentMatrix.isContinuous();					// matrix must be continous to use pointer. is not used in any other function though.

		if (continuousCheck){
			while (counter < medianFilterIterations){					// while iterations haven't been met.
				for (int y = 1; y < height - 1; y++){					// for each pixel.
					for (int x = 1; x < width - 1; x++) {
						int	kernelsum = 0;

						for (int i = 0; i < 9; i++){					// for that pixel and neighbours
							int tempY = i / 3 - 1 + y;
							int tempX = i % 3 - 1 + x;
							int valueIn = getPointer(tempX, tempY);
							kernelsum += valueIn;					// sum the numbers

							if (i == 8){									// when all the numbers have been summated
								if (kernelsum < 1275) temp.at<uchar>(tempY, tempX) = 0;			// if the sum suggest a majority of black pixels the center becomes black
								else temp.at<uchar>(y, x) = 255;					// else the center pixel becomes white
							}
						}
					}
				}
				segmentMatrix = temp.clone();											// update segment matrix
				counter++;													// increment how many times median has been applied
			}
		}
	}
	uchar getPointer(int x, int y){				// method of a pointer to a x, y position in a matrix.
		uchar* ptr = (uchar*)(segmentMatrix.data);
		uchar val = ptr[segmentMatrix.step * y + x];
		return val;
	}
};
class Blob{
private:
	Mat checkMatrix;
	int minX;
	int minY;
	int maxX;
	int maxY;
	int whitePixels;
	int expectedBlobs;
	unsigned int blobSize;
	unsigned int numberOfBlobs = 0;
	bool check = false;
	Point2i centerCoordinate;
	vector<Point2i> blobCoordinates;
	vector<int> blobSizes;
public:
	vector<Point2i> getBlobCoordinates(Mat mat, int num){
		expectedBlobs = num;
		checkMatrix = mat;

		if (whiteCheck()){								// checks if the number of white pixels in the image is not to large as it causes an overflow exception and breaks
			reset();
			for (size_t y = 0; y < checkMatrix.rows; y++) {				
				for (size_t x = 0; x < checkMatrix.cols; x++){			// check bounds.
					if (getPointToMatrix(x, y) == 255){			// if a pixel is white...
						initializeBlob(x, y);				// initialize blob...
						connectivity(x, y);				// start connectivity...
						addBlobCoordinate();				// add the blob coordinate in the vector of blobs
					}
				}
			}
			sort();									// bubblesort so that the largest blob is the first in the vector
		}
		fitToSize();									// resize the vector to fit the number of expected blobs 
		printData();									// print data about the blobs
		return blobCoordinates;								// return the vector with the blob coordinates
	}
protected:
	void connectivity(int x, int y){
		burn(x, y);																		// start by setting the pixel to be black.
		setCenterCoordinate(x, y);														// find the center coordinate of the blob.

		for (int i = 0; i < 4; i++){													// for the pixels in the connectivity kernel...
			int kernel[4][2] = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };
			int tempX = x + kernel[i][0];
			int tempY = y + kernel[i][1];
			int value = getPointToMatrix(tempX, tempY);
			if (value == 255) {															// if a pixel is white...
				connectivity(tempX, tempY);												// call the connectivity from the new x and y.
			}
		}
	}
	void reset(){																
		centerCoordinate.x = 0;															// reset the center coodinates.
		centerCoordinate.y = 0;
		numberOfBlobs = 0;																// reset the number of blobs.
		blobSize = 0;																	// reset the blob size.
		blobCoordinates.~vector();														// destruct the blobCoordinates vector.
		blobSizes.~vector();															// destruc the blobSizes vector.
	}
	void burn(int x, int y){	
		checkMatrix.at<uchar>(y, x) = 0;												// burn the pixel.
		blobSize++;																		// increment blob size.
	}
	void initializeBlob(int x, int y){
		blobSize = 0;																	// reset blob size.
		minX = checkMatrix.cols + 1;													// reset min and max values so that the first x and y always will be set to the new min x and y.
		minY = checkMatrix.rows + 1;
		maxX = -1;
		maxY = -1;
		numberOfBlobs++;										// increment the number of blobs
	}
	void addBlobCoordinate(){
		blobCoordinates.push_back(centerCoordinate);							// push back the center coordinate of the blob in the blob coordinates vector
		blobSizes.push_back(blobSize);									// push back the size of the blob in the blob size vector
	}
	void printData(){
		cout << "Expected BLOBs: " << expectedBlobs << endl;						// print expected blobs
		cout << "Number of BLOBs detected: " << numberOfBlobs << endl;					// print detected blobs
		if (blobCoordinates.size() > 0){								// if the size of the blob coordinate vector is greater than zero...
			cout << "Size of biggest BLOB: " << blobSizes.at(0) << endl;				// print size of the largest blob
			cout << "List of BLOB coordinates: " << blobCoordinates << endl;			// print the list of blobs
		}
		cout << endl;
	}
	void sort(){												
		bool flag = true;
		int start = expectedBlobs;
		int end = blobCoordinates.size() - 1;

		if (blobCoordinates.size() > expectedBlobs){
			while (flag){
				flag = false;

				for (int i = blobCoordinates.size() - 2; i >= 0; i--){				// go through the elements in the vector backwards
					if (blobSizes.at(i + 1) > blobSizes.at(i)){				// compare current element with next element
						int tempSize = blobSizes.at(i);					// copy current size and coordinates to temp (Swap step 1)
						Point2i tempCoordinate = blobCoordinates.at(i);			
						blobSizes.at(i) = blobSizes.at(i + 1);				// copy next size and coordinates to current (Swap step 2)
						blobCoordinates.at(i) = blobCoordinates.at(i + 1);		
						blobSizes.at(i + 1) = tempSize;					// copy temp size and coordinates to next (Swap step 3)
						blobCoordinates.at(i + 1) = tempCoordinate;				
						flag = true;							// flag=true to signify a swap has taken place
					}
				}
			}
		}
	}
	void fitToSize(){
		vector<Point2i> temp;

		if (blobCoordinates.size() > expectedBlobs){				// if the number of blob coordinates is greater than the expected blobs...
			for (int i = 0; i < expectedBlobs; i++){			// fit the blobCoordinates vector to be the size of the expected blobs
				temp.push_back(blobCoordinates.at(i));
			}
			blobCoordinates = temp;									
		}
		else if (blobCoordinates.size() < expectedBlobs){			// if the number of blob coordinates is less than the expected blobs...
			blobCoordinates.resize(expectedBlobs);				// resize the vectors to be the size of the expected blobs...
			blobSizes.resize(expectedBlobs);				// the added entries will be 0.
		}
		temp.~vector();								// destruct  the temp vector
	}
	uchar getPointToMatrix(int x, int y){						// gets a pointer to the matrix.
		if (x < 0) x = 0;							// Boundaries
		else if (x > 640) x = 640;
		if (y < 0) y = 0;
		else if (y > 480) y = 480;
		uchar* ptr = (uchar*)checkMatrix.data;
		uchar val = ptr[checkMatrix.step * y + x];
		return val;								// and return the value
	}
	void setCenterCoordinate(int x, int y){
		if (x > maxX) maxX = x;							// find the max and min values for x.
		else if (x < minX) minX = x;
		if (y > maxY) maxY = y;							// find the max and min values for y.
		else if (y < minY) minY = y;
		centerCoordinate.x = ((maxX - minX) / 2) + minX;			// set center x.
		centerCoordinate.y = ((maxY - minY) / 2) + minY;			// set center y.
	}
	bool whiteCheck(){
		whitePixels = 0;

		for (size_t y = 0; y < checkMatrix.rows; y++) {				// for all pixels in the image
			for (size_t x = 0; x < checkMatrix.cols; x++){
				uchar* ptr = (uchar*)(checkMatrix.data);			
				uchar val = ptr[checkMatrix.step * y + x];
				if (val == 255)	whitePixels++;				// if the pixel is white, increment white pixels
			}
		}
		if (whitePixels > 640 * 480 / 20) check = false;			// if a 1/20 of the pixels are white, the boolean becomes false
		else check = true;
		return check;
	}
};
class Tiles{
private:
	vector<Point2i> mainTiles;
	vector<Point2i> tiles;
	vector<Point2i> prevTiles;
	int tilesPerLine = 5;

public:
	void createTiles(vector<Point2i> vector){
		mainTiles = vector;
		tiles.resize(tilesPerLine * 3);
		Point2i temp;
		bool flag = true;
		prevTiles = tiles;

		if (vector.size() == 3){								// sort so that entry 0 becomes the greatest x
			while (flag) {									
				flag = false;
				for (int i = 0; i < 2; i++) {						// for every element in the vector
					if (mainTiles.at(i).x < mainTiles.at(i + 1).x) {		// compare current element with next element in vector
						temp.x = mainTiles.at(i).x;				// copy current coordinates to temp (Swap step 1)
						temp.y = mainTiles.at(i).y;											
						mainTiles.at(i).x = mainTiles.at(i + 1).x;		// copy next coordinates to current (Swap step 2)
						mainTiles.at(i).y = mainTiles.at(i + 1).y;							
						mainTiles.at(i + 1).x = temp.x;				// copy temp coordinates to next (Swap step 3)
						mainTiles.at(i + 1).y = temp.y;										
						flag = true;						// set flag to true, to signify that a switch took place
					}
				}
			}

			tiles.at(0) = mainTiles.at(1);							// the entries of the main tiles
			tiles.at(tilesPerLine) = mainTiles.at(0);
			tiles.at(tilesPerLine * 2) = mainTiles.at(2);
			int yMin = mainTiles.at(1).y;
			int height = mainTiles.at(2).y - mainTiles.at(1).y;

			for (int i = 1; i < tilesPerLine; i++){						// for tiles 2,3,4,5
				int width = mainTiles.at(0).x - mainTiles.at(1).x;
				tiles.at(i).x = width / tilesPerLine * i + mainTiles.at(1).x;
				tiles.at(i).y = height / tilesPerLine * i + yMin;
			}
			for (int i = 1 + tilesPerLine; i < tilesPerLine * 2; i++){			// for tiles 7,8,9,10
				int width = mainTiles.at(0).x - mainTiles.at(2).x;
				tiles.at(i).x = width / tilesPerLine * (i - tilesPerLine) + mainTiles.at(2).x;
				tiles.at(i).y = mainTiles.at(2).y;
			}
			for (int i = 1 + tilesPerLine * 2; i < tilesPerLine * 3; i++){			// for tiles 12,13,14,15
				int width = mainTiles.at(1).x - mainTiles.at(2).x;
				tiles.at(i).x = width / tilesPerLine * (tilesPerLine * 3 - i) + mainTiles.at(2).x;
				tiles.at(i).y = height / tilesPerLine * (i - tilesPerLine * 2) + yMin;
			}
		}
		else{											// if the size of the vector with the main tiles is not = 3...
			cout << "Not enough coordinates for tiles to be created." << endl;
			cout << endl;
			if (prevTiles.size() == 15)	tiles = prevTiles;				// if the size of the prevTiles is = 15, tiles = previous tiles
		}
	}
	vector<Point2i> getTiles(){
		return tiles;										// returns tiles
	}
	void projectTiles(int projectedTiles[], Mat frame){
		int spacing = 10;
		int r, g, b;
		Point2i point1, point2;
		Mat projectedBoard = Mat::zeros(frame.size(), frame.type());
		projectedBoard = Scalar(255, 255, 255);							// makes the matrix white

		for (int i = 0; i < tiles.size(); i++){							// for all the tiles
			r = 0;
			g = 0;
			b = 0;
			// if (i == projectedTiles[0]...						// projected tiles should be coloured. not implemented!!!
			if (i == 0) {																			
				r = 255;								// tile 0 becomes red.
			}
			if (i == 5) {
				b = 255;								// tile 5 becomes blue.
			}
			if (i == 10) {																			
				g = 255;								// tile 10 becomes green.
			}
			point1.x = tiles.at(i).x - spacing;
			point1.y = tiles.at(i).y - spacing;
			point2.x = tiles.at(i).x + spacing;
			point2.y = tiles.at(i).y + spacing;
			rectangle(projectedBoard, point1, point2, CV_RGB(r, g, b), 1);			// makes a rectangle for each of the tiles.
		}
		imshow("Board", projectedBoard);							// show the projected Board.
	}
}tiles;
int main(int, char)
{
	/* 	Declare matrices.
	Declare and initialise the objects for segmenting, pieces and blobs.
	Segmentation objects are used for segmenting the camera input and have their own thresholding values. These values are not fully calibrated.
	Pieces objects handles the position of the piece. It used the tile coordinates to specify which tile the piece is on.
	Blob objects uses connectivity to find the center coordinate of the blobs of the segmented images.*/

	Mat src;
	Mat frame;

	Segmentation redPieceSegment(111, 123, 2, 2);
	Segmentation greenPieceSegment(46, 50, 2, 2);
	Segmentation bluePieceSegment(0, 10, 2, 2);
	Segmentation diceSegmented(118, 180, 1, 1);
	Segmentation tilesSegment(230, 255, 2, 2);
	Segmentation dice(1, 1, 1, 1);
	Piece redPiece;
	Piece greenPiece;
	Piece bluePiece;
	Blob greenBlob;
	Blob redBlob;
	Blob blueBlob;
	Blob tilesBlob;

	VideoCapture cap(0);
	cap >> src;
	if (!cap.isOpened())
		return -1;
	else {

		for (;;) {

			/*	Get the frame, flip it the correct way, convert to HSV colourspace.	*/
			cap >> frame;
			flip(frame, frame, 1);
			cvtColor(frame, src, CV_RGB2HSV, 3);

			/*	Segment the pieces and the main tiles. Threshold on hue for the pieces and threshold on value for the tiles. */
			Mat redPieceSegmented = redPieceSegment.segmentHue(src);
			//	Mat greenPieceSegmented = greenPieceSegment.segmentHue(src);	
			//	Mat bluePieceSegmented = bluePieceSegment.segmentHue(src);																								
			//	Mat segmentedTiles = tilesSegment.segmentValue(src);
			//	Mat segmentedDice = diceSegmented.segmentHue(src);

			/*	The rest of the code has not been fully tested and is therefor commented out.
			Update the dice by passing the segmented matrix for the dice. */
			//	int diceRoll = dice.update(segmentedDice.segmentHue(src));

			/*	Update the board. Create the tiles by using the coordinates from the segmented tiles image.
			update the position of the pieces by using tile coordinates and piece coordinates. */
			//	tiles.createTiles(tilesBlob.getBlobCoordinates(segmentedTiles, 3));	
			//	redPiece.updatePos(tiles.getTiles(), redBlob.getBlobCoordinates(redPieceSegmented, 1));											
			//	greenPiece.updatePos(tiles.getTiles(), greenBlob.getBlobCoordinates(greenPieceSegmented, 1));
			//	bluePiece.updatePos(tiles.getTiles(), blueBlob.getBlobCoordinates(bluePieceSegmented, 1));

			/*	If the blob coordinates were found for the pieces, find the projection tiles and update the image showing the board.*/
			//	if (redPiece.wasDetected() && greenPiece.wasDetected() && bluePiece.wasDetected()){
			//		int projectedTiles[3] = { redPiece.getProjectionTile(dice.getDiceRoll()), greenPiece.getProjectionTile(dice.getDiceRoll()), bluePiece.getProjectionTile(dice.getDiceRoll()) };
			//		tiles.projectTiles(projectedTiles, src);
			//	}

			/*	Will print the blob data for the red blob and show the blob in a window. */
			imshow("Red Piece Blobs", redPieceSegmented);													// should be switched with the line above.
			redBlob.getBlobCoordinates(redPieceSegmented, 3);

			if (waitKey(30) >= 0)
				break;
		}
	}
	return 0;
}
