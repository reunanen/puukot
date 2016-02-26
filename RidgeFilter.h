#include <opencv2/core/core.hpp>

struct RidgeFilterParameters {
    double alpha = 0.01; // increase to react more rapidly
    double beta = 0.99; // usually 1.0 - alpha
    int dilationWidth = 1; // usually 1 or 2
    int suppressionWidth = 48;
    int outputType = -1; // -1: use input type
    int accumulatorType = CV_64FC1;
    double outputGain = 1.0;
};

// Helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct RidgeFilterSingleDirectionTemp
{
    cv::Mat accumulator;
    cv::Mat suppressedAccumulator;
};

// Another helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct RidgeFilterBidirectionalTemp {
    cv::Mat temp1;
    cv::Mat temp2;
    RidgeFilterSingleDirectionTemp singleDirectionTemp;
};

void RidgeFilterDown(const cv::Mat& input, cv::Mat& output, const RidgeFilterParameters& parameters, RidgeFilterSingleDirectionTemp& temp = RidgeFilterSingleDirectionTemp());
void RidgeFilterUpDown(const cv::Mat& input, cv::Mat& output, const RidgeFilterParameters& parameters, RidgeFilterBidirectionalTemp& temp = RidgeFilterBidirectionalTemp());
