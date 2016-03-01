#include "../../RidgeFilter.h"
#include "../../HighpassFilter.h"
#include "../../TwoLevelDetection.h"
#include "opencv-show-pixel-value/show-pixel-value-connecter.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat GenerateRidgeFilterInputImage()
{
    cv::Mat inputImage(600, 800, CV_8UC1);
    cv::randn(inputImage, 160, 32);

    cv::Mat lineImage(inputImage.size(), inputImage.type());
    lineImage.setTo(0);

    std::vector<cv::Point> points = {
        cv::Point(100, 100),
        cv::Point(300, 500),
        cv::Point(500, 100),
        cv::Point(700, 300),
        cv::Point(600, 500),
    };
    cv::polylines(lineImage, points, false, 8, 3, CV_AA);
    cv::polylines(lineImage, points, false, 24, 1, CV_AA);

    inputImage += lineImage;

    cv::circle(inputImage, cv::Point(300, 100), 3, 255, -1);

    return inputImage;
}

void RunRidgeFilterTest()
{
    cv::Mat inputImage = GenerateRidgeFilterInputImage();

    RidgeFilterParameters ridgeFilterParameters;
    ridgeFilterParameters.dilationWidth = 2;
    ridgeFilterParameters.outputGain = 32;

    cv::Mat ridgeFilteringResult;
    RidgeFilterUpDown(inputImage, ridgeFilteringResult, ridgeFilterParameters);

    cv::Mat highpassFilteredRidgeResult;
    HighpassFilter(ridgeFilteringResult, highpassFilteredRidgeResult, HighpassFilterParameters(cv::Size(80, 25), CV_8UC1));

    cv::Mat detectionResult;
    TwoLevelDetection(highpassFilteredRidgeResult, detectionResult, 50, 20);

    cv::Mat resultImage;
    cv::cvtColor(inputImage, resultImage, cv::COLOR_GRAY2BGR);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(detectionResult, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(resultImage, contours, -1, cv::Scalar(0, 0, 255));

    ShowPixelValueConnecter connecter;
    connecter
        .Add("Input", inputImage)
        .Add("Ridge filtering result", ridgeFilteringResult)
        .Add("Highpass filtered ridge result", highpassFilteredRidgeResult)
        .Add("Detection result", resultImage);

    cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    RunRidgeFilterTest();

    return 0;
}
