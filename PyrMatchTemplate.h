#include <opencv2/core/core.hpp>
#include <deque>

struct PyrMatchTemplateParameters {
    size_t maxPyramidLevels = 2; // TODO: maybe 5?
    int minTemplateLesserDimension = 5;
};

// Helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct PyrMatchTemplateTemp {
    std::deque<cv::Mat> inputPyramid;
    std::deque<cv::Mat> templPyramid;
    std::deque<cv::Mat> outputPyramid;
    std::deque<cv::Point> offsets;
};

// TODO: change to cv::InputArray and cv::OutputArray
void PyrMatchTemplate(const cv::Mat& input, const cv::Mat& templ, cv::Mat& result, int method, const PyrMatchTemplateParameters* parameters = nullptr, PyrMatchTemplateTemp* temp = nullptr);
