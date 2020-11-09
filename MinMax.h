#include <opencv2/core/core.hpp>

struct MinMax
{
    double minValue = std::numeric_limits<double>::quiet_NaN();
    double maxValue = std::numeric_limits<double>::quiet_NaN();
    cv::Point minLocation;
    cv::Point maxLocation;
};

MinMax FindMinMax(cv::InputArray input, cv::InputArray mask = cv::noArray());
