#include <opencv2/core/core.hpp>

// To be detected, at least one pixel in each blob must exceed detectionLevel.
// However, many more pixels can remain in each result blob - namely, those that exceed measurementLevel.
// In other words, measurementLevel needs to be lower than detectionLevel (or equal, but then the method
// reduces to plain thresholding).

struct TwoLevelDetectionParameters {
    TwoLevelDetectionParameters(double detectionLevel, double measurementLevel, size_t minContourLength = 0)
        : detectionLevel(detectionLevel)
        , measurementLevel(measurementLevel)
        , minContourLength(minContourLength)
    {}

    double detectionLevel;
    double measurementLevel;
    size_t minContourLength;
};

// Helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct TwoLevelDetectionTemp {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat mask;
};

void TwoLevelDetection(const cv::Mat& input, cv::Mat& output, const TwoLevelDetectionParameters& parameters, TwoLevelDetectionTemp& temp = TwoLevelDetectionTemp());

// A convenience wrapper for backward compatibility. TODO: remove this.
void TwoLevelDetection(const cv::Mat& input, cv::Mat& output, double detectionLevel, double measurementLevel, TwoLevelDetectionTemp& temp = TwoLevelDetectionTemp());
