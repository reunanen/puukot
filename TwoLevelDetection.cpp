#include "TwoLevelDetection.h"
#include <opencv2/imgproc/imgproc.hpp>

void TwoLevelDetection(const cv::Mat& input, cv::Mat& output, const TwoLevelDetectionParameters& parameters, TwoLevelDetectionTemp& temp)
{
    if (parameters.measurementLevel > parameters.detectionLevel) {
        throw std::runtime_error("TwoLevelDetection: it is required that measurementLevel <= detectionLevel");
    }

    cv::threshold(input, output, parameters.measurementLevel, 1, cv::THRESH_BINARY);

    temp.contours.clear();
    cv::findContours(output, temp.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    temp.mask.create(input.size(), CV_8UC1);

    output.setTo(0);

    for (size_t i = 0, end = temp.contours.size(); i < end; ++i) {
        if (temp.contours[i].size() >= parameters.minContourLength) {
            temp.mask.setTo(0);
            cv::drawContours(temp.mask, temp.contours, static_cast<int>(i), 255, -1);

            double minVal, maxVal;
            cv::minMaxIdx(input, &minVal, &maxVal, NULL, NULL, temp.mask);

            if (maxVal >= parameters.detectionLevel) {
                output.setTo(std::numeric_limits<unsigned char>::max(), temp.mask);
            }
        }
    }
}

// A convenience wrapper for backward compatibility. TODO: remove this.
void TwoLevelDetection(const cv::Mat& input, cv::Mat& output, double detectionLevel, double measurementLevel, TwoLevelDetectionTemp& temp)
{
    TwoLevelDetection(input, output, TwoLevelDetectionParameters(detectionLevel, measurementLevel), temp);
}