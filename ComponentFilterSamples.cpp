#include "ComponentFilterSamples.h"
#include <opencv2/imgproc/imgproc.hpp>

// This is a sample criterion: minimum contour length.
bool MinimumContourLengthCriterion::operator()(ComponentFilterCriterionInput& input) const
{
    return input.contour.size() >= minimumContourLength;
}

// Another sample criterion: minimum area.
bool MinimumAreaCriterion::operator()(ComponentFilterCriterionInput& input) const
{
    temp.create(input.inputImage.size(), CV_8UC1);
    temp.setTo(0);
    cv::drawContours(temp, std::vector<std::vector<cv::Point>>({ input.contour }), 0, 255, -1);
    return cv::countNonZero(temp) >= minimumArea;
}

// Adapted from: http://stackoverflow.com/questions/14854592/retrieve-elongation-feature-in-python-opencv-what-kind-of-moment-it-supposed-to
double CalculateElongation(const cv::Moments& moments)
{
    const double x = moments.mu20 + moments.mu02;
    const double y = 4 * pow(moments.mu11, 2) + pow(moments.mu20 - moments.mu02, 2);
    return (x + sqrt(y)) / (x - sqrt(y));
}

// Yet another sample: minimum elongation.
bool MinimumElongationCriterion::operator()(ComponentFilterCriterionInput& input) const
{
    temp.create(input.inputImage.size(), CV_8UC1);
    temp.setTo(0);
    cv::drawContours(temp, std::vector<std::vector<cv::Point>>({ input.contour }), 0, 255, -1);
    const double elongation = CalculateElongation(cv::moments(temp));
    return elongation >= minimumElongation;
}
