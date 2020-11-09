#include "PyrMatchTemplate.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>

void PyrMatchTemplate(const cv::Mat& input, const cv::Mat& templ, cv::Mat& result, int method, const PyrMatchTemplateParameters* parameters, PyrMatchTemplateTemp* temp)
{
    // First, some preparations

    std::unique_ptr<PyrMatchTemplateParameters> defaultParameters;
    if (!parameters) {
        defaultParameters = std::make_unique<PyrMatchTemplateParameters>();
        parameters = defaultParameters.get();
    }

    std::unique_ptr<PyrMatchTemplateTemp> myTemp;
    if (!temp) {
        myTemp = std::make_unique<PyrMatchTemplateTemp>();
        temp = myTemp.get();
    }

    assert(parameters != nullptr);
    assert(temp != nullptr);

    // Now we have valid parameters and the "temp" object, so let's get to the actual subject matter

    const auto pushOrSet = [](std::deque<cv::Mat>& pyramid, cv::Mat image, size_t index) {
        if (index > pyramid.size()) {
            throw std::runtime_error("Unexpected index relative to current pyramid size");
        }
        if (index == pyramid.size()) {
            pyramid.push_back(image);
        }
        else {
            pyramid[index] = image;
        }
    };

    size_t pyramidLevel = 0;

    pushOrSet(temp->inputPyramid, input, pyramidLevel);
    pushOrSet(temp->templPyramid, templ, pyramidLevel);

    const auto getCurrentTemplateLesserDimension = [&pyramidLevel, &temp]() {
        const auto currentTemplate = temp->templPyramid[pyramidLevel];
        return std::min(
            currentTemplate.rows,
            currentTemplate.cols
        );
    };

    const auto pyrDownscale = [](int dimension) {
        return (dimension + 1) / 2;
    };

    const auto pyrUpscale = [](int dimension) {
        return dimension * 2; // - 1;
    };

    while (pyrDownscale(getCurrentTemplateLesserDimension()) > parameters->minTemplateLesserDimension && pyramidLevel < parameters->maxPyramidLevels) {
        ++pyramidLevel;

        const auto makeRoomIfNeeded = [pyramidLevel](std::deque<cv::Mat>& pyramid) {
            if (pyramidLevel >= pyramid.size()) {
                pyramid.emplace_back();
            }
        };

        makeRoomIfNeeded(temp->inputPyramid);        
        makeRoomIfNeeded(temp->templPyramid);

        cv::pyrDown(temp->inputPyramid[pyramidLevel - 1], temp->inputPyramid[pyramidLevel]);
        cv::pyrDown(temp->templPyramid[pyramidLevel - 1], temp->templPyramid[pyramidLevel]);
    }

    if (temp->outputPyramid.size() <= pyramidLevel) {
        temp->outputPyramid.resize(pyramidLevel + 1);
    }
    if (temp->offsets.size() <= pyramidLevel) {
        temp->offsets.resize(pyramidLevel + 1);
    }

    //temp->offsets[pyramidLevel] = cv::Point(0, 0);

    result.create(input.rows - templ.rows + 1, input.cols - templ.cols + 1, CV_32FC1);

    result = (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED)
        ? std::numeric_limits<float>::max()
        : std::numeric_limits<float>::lowest();

    cv::Rect resultRect(0, 0, result.cols, result.rows);
    cv::Rect inputRect(0, 0, temp->inputPyramid[pyramidLevel].cols, temp->inputPyramid[pyramidLevel].rows);

    const size_t maxPyramidLevel = pyramidLevel;

    // TODO: copy only the result of pyramid level 0 to the output! otherwise the output should be +/- infinity (depending on method), or something like that
    //       ... or, if it can be made _valid_, then that'd be even better of course

    while (true) {
        cv::matchTemplate(temp->inputPyramid[pyramidLevel](inputRect), temp->templPyramid[pyramidLevel], temp->outputPyramid[pyramidLevel], method);

        assert(pyramidLevel == maxPyramidLevel || temp->outputPyramid[pyramidLevel].size() == resultRect.size());

        const double scaler = pow(2, pyramidLevel);

        if (method == cv::TM_SQDIFF || method == cv::TM_CCORR || method == cv::TM_CCOEFF) {
            temp->outputPyramid[pyramidLevel] *= scaler * scaler;
        }

        if (pyramidLevel == 0) {
            assert(temp->outputPyramid[pyramidLevel].size() == resultRect.size());
            temp->outputPyramid[pyramidLevel].copyTo(result(resultRect));
            //cv::resize(temp->outputPyramid[pyramidLevel], result(resultRect), resultRect.size(), 0.0, 0.0, cv::INTER_NEAREST);
            break;
        }

        double minValue = std::numeric_limits<double>::quiet_NaN();
        double maxValue = std::numeric_limits<double>::quiet_NaN();
        cv::Point minLocation, maxLocation;
        cv::minMaxLoc(temp->outputPyramid[pyramidLevel], &minValue, &maxValue, &minLocation, &maxLocation);

        // TODO: support several top locations (not just one)

        const cv::Point location = (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED)
            ? minLocation
            : maxLocation;

        cv::Rect bestMatch(
            inputRect.x + location.x,
            inputRect.y + location.y,
            temp->templPyramid[pyramidLevel].cols,
            temp->templPyramid[pyramidLevel].rows
        );

        // TODO: add some parameters here
        const int w = 10; // temp->templPyramid[pyramidLevel].cols;
        const int h = 10; // temp->templPyramid[pyramidLevel].rows;

        inputRect = cv::Rect(
            bestMatch.x - w,
            bestMatch.y - h,
            bestMatch.width  + 2 * w,
            bestMatch.height + 2 * h
        );

        inputRect &= cv::Rect(0, 0, temp->inputPyramid[pyramidLevel].cols, temp->inputPyramid[pyramidLevel].rows);

        // hmm something wrong here probably
        inputRect.x = pyrUpscale(inputRect.x);
        inputRect.y = pyrUpscale(inputRect.y);
        inputRect.width  = pyrUpscale(inputRect.width);
        inputRect.height = pyrUpscale(inputRect.height);

        resultRect.width  = inputRect.width  - temp->templPyramid[pyramidLevel - 1].cols + 1;
        resultRect.height = inputRect.height - temp->templPyramid[pyramidLevel - 1].rows + 1;
        resultRect.x = static_cast<int>(std::round(bestMatch.x * scaler - resultRect.width  / 2));
        resultRect.y = static_cast<int>(std::round(bestMatch.y * scaler - resultRect.height / 2));

        resultRect &= cv::Rect(0, 0, result.cols, result.rows);

        --pyramidLevel;
    }
}
