#include "ComponentFilter.h"
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _DEBUG
#include <iterator> // std::inserter
#endif

ComponentFilterCriterionInput::ComponentFilterCriterionInput(const cv::Mat& inputImage, const std::vector<cv::Point>& contour)
    : inputImage(inputImage), contour(contour)
{}

cv::Mat DecideFindContoursTemp(const cv::Mat& input, cv::Mat& output, ComponentFilterTemp& temp)
{
    if (input.data == output.data) {
        // In-place operation
        input.copyTo(temp.findContoursTemp);
        return temp.findContoursTemp;
    }
    else {
        input.copyTo(output);
        return output;
    }
}

unsigned int ComponentFilter(const cv::Mat& input, cv::Mat& output, const ComponentFilterCriterion& criterion, ComponentFilterTemp& temp)
{
    cv::Mat findContoursTemp = DecideFindContoursTemp(input, output, temp);
    cv::findContours(findContoursTemp, temp.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

#ifdef _DEBUG
    cv::Mat inputWithHolesFilled(input.size(), CV_8UC1);
    inputWithHolesFilled.setTo(0);
    cv::drawContours(inputWithHolesFilled, temp.contours, -1, std::numeric_limits<unsigned char>::max(), -1);
#endif

    output.setTo(0);
    unsigned int acceptedComponents = 0;
    for (size_t i = 0, end = temp.contours.size(); i < end; ++i) {
        ComponentFilterCriterionInput componentFilterCriterionInput(input, temp.contours[i]);
        if (criterion(componentFilterCriterionInput)) {
            cv::drawContours(output, temp.contours, static_cast<int>(i), std::numeric_limits<unsigned char>::max(), -1);
            ++acceptedComponents;
        }
    }

#ifdef _DEBUG
    // We can only filter - we can't introduce any new components. Ever.
    std::vector<cv::Point> inputPoints, outputPoints, outputPointsLessInputPoints;
    if (cv::countNonZero(inputWithHolesFilled)) {
        cv::findNonZero(inputWithHolesFilled, inputPoints);
    }
    if (cv::countNonZero(output)) {
        cv::findNonZero(output, outputPoints);
    }
    const auto sortCriterion = [](const cv::Point& point1, const cv::Point& point2) {
        return point1.x < point2.x || ((point1.x == point2.x && point1.y < point2.y));
    };
    std::sort(inputPoints.begin(), inputPoints.end(), sortCriterion);
    std::sort(outputPoints.begin(), outputPoints.end(), sortCriterion);
    std::set_difference(outputPoints.begin(), outputPoints.end(), inputPoints.begin(), inputPoints.end(), std::inserter(outputPointsLessInputPoints, outputPointsLessInputPoints.begin()), sortCriterion);
    assert(outputPointsLessInputPoints.empty());
#endif

    return acceptedComponents;
}
