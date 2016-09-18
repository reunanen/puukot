#include "TwoLevelDetection.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace {
    bool AnyFindings(const cv::Mat& input, TwoLevelDetectionTemp& temp) {
        if (!temp.zeroRow.data || temp.zeroRow.size() != input.row(0).size() || temp.zeroRow.type() != input.type()) {
            temp.zeroRow.create(input.row(0).size(), input.type());
            temp.zeroRow.setTo(0);
        }
        const void* zeroRowPtr = temp.zeroRow.ptr(0);
        const size_t rowLengthInBytes = input.cols * input.elemSize();
        for (int row = 0; row < input.rows; ++row) {
            const void* rowPtr = input.ptr(row);
            if (memcmp(rowPtr, zeroRowPtr, rowLengthInBytes) != 0) {
                return true;
            }
        }
        return false;
    }
}

unsigned int TwoLevelDetection(const cv::Mat& input, cv::Mat& output, const TwoLevelDetectionParameters& parameters, TwoLevelDetectionTemp& temp)
{
    if (parameters.measurementLevel > parameters.detectionLevel) {
        throw std::runtime_error("TwoLevelDetection: it is required that measurementLevel <= detectionLevel");
    }

    if (input.data == output.data) {
        throw std::runtime_error("TwoLevelDetection: in-place operation not supported");
    }

    cv::threshold(input, temp.mask, parameters.detectionLevel, 1, cv::THRESH_BINARY);
    if (!AnyFindings(temp.mask, temp)) {
        // We can't possibly have anything to return, so let's just take a quick way out.
        output.create(input.size(), CV_8UC1);
        output.setTo(0);
        return 0;
    }

    cv::threshold(input, output, parameters.measurementLevel, 1, cv::THRESH_BINARY);

    unsigned int regionsAccepted = 0;

    temp.detectionSeeds.clear();
    cv::findNonZero(temp.mask, temp.detectionSeeds);

    for (const cv::Point& seedPoint : temp.detectionSeeds) {
        unsigned char outputValue = output.at<unsigned char>(seedPoint);
        assert(outputValue == 1 || outputValue == 128);
        if (outputValue == 1) {
            cv::floodFill(output, seedPoint, 128);
                ++regionsAccepted;
            }
        }

    cv::threshold(output, output, 127, std::numeric_limits<unsigned char>::max(), cv::THRESH_BINARY);

    return regionsAccepted;
}

// A convenience wrapper for backward compatibility. TODO: remove this.
unsigned int TwoLevelDetection(const cv::Mat& input, cv::Mat& output, double detectionLevel, double measurementLevel, TwoLevelDetectionTemp& temp)
{
    return TwoLevelDetection(input, output, TwoLevelDetectionParameters(detectionLevel, measurementLevel), temp);
}