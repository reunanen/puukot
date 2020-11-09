#include "MinMax.h"
#include <opencv2/core/core.hpp>

MinMax FindMinMax(cv::InputArray input, cv::InputArray mask)
{
    MinMax minMax;
    
    cv::minMaxLoc(input, &minMax.minValue, &minMax.maxValue, &minMax.minLocation, &minMax.maxLocation, mask);

    return minMax;
}
