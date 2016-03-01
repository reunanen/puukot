#include "RidgeFilter.h"
#include <opencv2/imgproc/imgproc.hpp>

int DecideOutputType(int configuredOutputType, int inputType)
{
    if (configuredOutputType >= 0) {
        return configuredOutputType;
    }
    else {
        return inputType; // default: use input type
    }
}

void RidgeFilterDown(const cv::Mat& input, cv::Mat& output, const RidgeFilterParameters& parameters, RidgeFilterSingleDirectionTemp& temp)
{
    const int outputType = DecideOutputType(parameters.outputType, input.type());
    output.create(input.size(), outputType);

    temp.accumulator.create(cv::Size(input.cols, 1), parameters.accumulatorType);
    temp.suppressedAccumulator.create(cv::Size(input.cols, 1), parameters.accumulatorType);

    temp.accumulator.setTo(0);

    for (int y = 0; y < input.rows; ++y) {
        cv::addWeighted(input.row(y), parameters.alpha, temp.accumulator, parameters.beta, 0, temp.accumulator, parameters.accumulatorType);
        cv::dilate(temp.accumulator, temp.accumulator, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * parameters.dilationWidth + 1, 1)));
        cv::subtract(temp.accumulator, cv::mean(temp.accumulator)[0], temp.suppressedAccumulator);
        temp.suppressedAccumulator.convertTo(output.row(y), output.type(), parameters.outputGain);
    }
}

void RidgeFilterUp(const cv::Mat& input, cv::Mat& output, const RidgeFilterParameters& parameters, cv::Mat& temp, RidgeFilterSingleDirectionTemp& singleDirectionTemp)
{
    cv::Mat& inputUpsideDown = temp;
    cv::flip(input, inputUpsideDown, 0);
    RidgeFilterDown(inputUpsideDown, output, parameters, singleDirectionTemp);
    cv::flip(output, output, 0);
}

void RidgeFilterUpDown(const cv::Mat& input, cv::Mat& output, const RidgeFilterParameters& parameters, RidgeFilterBidirectionalTemp& temp)
{
    cv::Mat& up = temp.temp1;
    RidgeFilterUp(input, up, parameters, temp.temp2, temp.singleDirectionTemp);

    cv::Mat& down = temp.temp2;
    RidgeFilterDown(input, down, parameters, temp.singleDirectionTemp);

    cv::addWeighted(up, 0.5, down, 0.5, 0.0, output, DecideOutputType(parameters.outputType, input.type()));
}
