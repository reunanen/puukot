#include "ComponentFilter.h"
#include <opencv2/imgproc/imgproc.hpp>

ComponentFilterCriterionInput::ComponentFilterCriterionInput(const cv::Mat& inputImage, const std::vector<cv::Point>& contour)
    : inputImage(inputImage), contour(contour)
{}

void ComponentFilter(const cv::Mat& input, cv::Mat& output, const ComponentFilterCriterion& criterion, ComponentFilterTemp& temp)
{
    input.copyTo(output);
    cv::findContours(output, temp.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    output.setTo(0);
    for (size_t i = 0, end = temp.contours.size(); i < end; ++i) {
        ComponentFilterCriterionInput componentFilterCriterionInput(input, temp.contours[i]);
        if (criterion(componentFilterCriterionInput)) {
            cv::drawContours(output, temp.contours, static_cast<int>(i), std::numeric_limits<unsigned char>::max(), -1);
        }
    }

#ifdef _DEBUG
    // We can only filter - we can't introduce any new components. Ever.
    std::vector<cv::Point> inputPoints, outputPoints, outputPointsLessInputPoints;
    if (cv::countNonZero(input)) {
        cv::findNonZero(input, inputPoints);
    }
    if (cv::countNonZero(output)) {
        cv::findNonZero(output, outputPoints);
    }
    const auto sortCriterion = [](const cv::Point& point1, const cv::Point& point2) {
        return point1.x < point2.x || ((point1.x == point2.x && point1.y < point2.y));
    };
    std::sort(inputPoints.begin(), inputPoints.end(), sortCriterion);
    std::sort(outputPoints.begin(), outputPoints.end(), sortCriterion);
    std::set_difference(inputPoints.begin(), inputPoints.end(), outputPoints.begin(), outputPoints.end(), std::inserter(outputPointsLessInputPoints, outputPointsLessInputPoints.begin()), sortCriterion);
    assert(outputPointsLessInputPoints.empty());
#endif
}