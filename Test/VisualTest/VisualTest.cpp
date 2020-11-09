#include "../../PyrMatchTemplate.h"
#include "../../RidgeFilter.h"
#include "../../HighpassFilter.h"
#include "../../TwoLevelDetection.h"
#include "opencv-show-pixel-value/show-pixel-value-connecter.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <iostream>

cv::Mat GeneratePyrTemplateMatchInputImage()
{
    cv::Mat image(600, 800, CV_8UC1);
    cv::randn(image, 64, 32);

    cv::Mat circle(600, 800, CV_8UC1);
    circle = 0;
    cv::circle(circle, cv::Point(400, 300), 200, 128, cv::FILLED);

    image += circle;

    return image;
}

void RunPyrTemplateMatchTest()
{
    cv::Mat inputImage = GeneratePyrTemplateMatchInputImage();
    cv::Mat templ = inputImage(cv::Rect(500, 400, 100, 100));

    cv::Mat cvMatchTemplateResult, pyrMatchTemplateResult;

    int method = cv::TM_SQDIFF;

    const auto t1 = std::chrono::steady_clock::now();

    cv::matchTemplate(inputImage, templ, cvMatchTemplateResult, method);

    const auto t2 = std::chrono::steady_clock::now();

    PyrMatchTemplate(inputImage, templ, pyrMatchTemplateResult, method);

    const auto t3 = std::chrono::steady_clock::now();

    std::cout << "cv::matchTemplate : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    std::cout << "PyrMatchTemplate  : " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << std::endl;

    double minVal = 0, maxVal = 0;
    cv::minMaxLoc(cvMatchTemplateResult, &minVal, &maxVal);

    double alpha = 255.0 / (maxVal - minVal);
    double beta = -alpha * minVal;

    cv::threshold(alpha * cvMatchTemplateResult  + beta, cvMatchTemplateResult,  255.0, 255.0, cv::THRESH_TRUNC);
    cv::threshold(alpha * pyrMatchTemplateResult + beta, pyrMatchTemplateResult, 255.0, 255.0, cv::THRESH_TRUNC);

    cvMatchTemplateResult .convertTo(cvMatchTemplateResult,  CV_8UC1);
    pyrMatchTemplateResult.convertTo(pyrMatchTemplateResult, CV_8UC1);

    ShowPixelValueConnecter connecter;
    connecter
        .Add("Image", inputImage)
        .Add("Template", templ)
        .Add("cv::matchTemplate result", cvMatchTemplateResult)
        .Add("PyrMatchTemplate result", pyrMatchTemplateResult)
    ;

    cv::waitKey(0);
}

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
    cv::polylines(lineImage, points, false, 8, 3, cv::LINE_AA);
    cv::polylines(lineImage, points, false, 24, 1, cv::LINE_AA);

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
    RunPyrTemplateMatchTest();

    RunRidgeFilterTest();

    return 0;
}
