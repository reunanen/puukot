#include "HighpassFilter.h"
#include <opencv2/imgproc/imgproc.hpp>

HighpassFilterParameters::HighpassFilterParameters(const cv::Size& blurKernel, int outputType)
    : blurKernel(blurKernel)
    , configuredOutputType(outputType)
    , borderType(cv::BORDER_REPLICATE)
{
}

// TODO: Deduplicate w.r.t. DecideOutputType() in RidgeFilter.cpp
int DecideHighpassFilterOutputType(int configuredOutputType, int inputType)
{
    if (configuredOutputType >= 0) {
        return configuredOutputType;
    }
    else {
        return inputType; // default: use input type
    }
}

void HighpassFilter(const cv::Mat& input, cv::Mat& output, const HighpassFilterParameters& parameters, HighpassFilterTemp& temp)
{
    const int outputType = DecideHighpassFilterOutputType(parameters.configuredOutputType, input.type());

    if (outputType == input.type()) {
        temp.inputConverted = input;
    }
    else {
        input.convertTo(temp.inputConverted, outputType);
    }
    cv::blur(temp.inputConverted, temp.blurred, parameters.blurKernel, cv::Point(-1, -1), parameters.borderType);
    cv::subtract(temp.inputConverted, temp.blurred, output, cv::noArray(), outputType);
}