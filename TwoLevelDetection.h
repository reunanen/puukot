#ifndef PUUKOT_TWO_LEVEL_DETECTION
#define PUUKOT_TWO_LEVEL_DETECTION

#include <opencv2/core/core.hpp>

// To be detected, at least one pixel in each blob must exceed detectionLevel.
// However, many more pixels can remain in each result blob - namely, those that exceed measurementLevel.
// In other words, measurementLevel needs to be lower than detectionLevel (or equal, but then the method
// reduces to plain thresholding).
// If you are pretty sure there's something to be found, set findingsExpected = true to disable a
// shortcut in the processing. (Evaluating the shortcut is not free, which is why it can be disabled.)

struct TwoLevelDetectionParameters {
    TwoLevelDetectionParameters(double detectionLevel, double measurementLevel, bool findingsExpected = false)
        : detectionLevel(detectionLevel)
        , measurementLevel(measurementLevel)
        , findingsExpected(findingsExpected)
    {}

    double detectionLevel;
    double measurementLevel;
    bool findingsExpected;
};

// Helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct TwoLevelDetectionTemp {
    std::vector<cv::Point> detectionSeeds;
    cv::Mat mask;
    cv::Mat zeroRow;
};

// Returns the number of regions found.
unsigned int TwoLevelDetection(const cv::Mat& input, cv::Mat& output, const TwoLevelDetectionParameters& parameters, TwoLevelDetectionTemp& temp = TwoLevelDetectionTemp());

// A convenience wrapper for backward compatibility. TODO: remove this.
unsigned int TwoLevelDetection(const cv::Mat& input, cv::Mat& output, double detectionLevel, double measurementLevel, TwoLevelDetectionTemp& temp = TwoLevelDetectionTemp());

#endif // PUUKOT_TWO_LEVEL_DETECTION