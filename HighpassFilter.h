#include <opencv2/core/core.hpp>

struct HighpassFilterParameters {
    HighpassFilterParameters(const cv::Size& blurKernel, int outputType = -1);

    cv::Size blurKernel;
    int configuredOutputType;
    int borderType;
};

// Helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct HighpassFilterTemp
{
    cv::Mat inputConverted;
    cv::Mat blurred;
};

void HighpassFilter(const cv::Mat& input, cv::Mat& output, const HighpassFilterParameters& parameters, HighpassFilterTemp& temp = HighpassFilterTemp());
