#ifndef PUUKOT_COMPONENT_FILTER
#define PUUKOT_COMPONENT_FILTER

#include <opencv2/core/core.hpp>

struct ComponentFilterCriterionInput
{
    ComponentFilterCriterionInput(const cv::Mat& inputImage, const std::vector<cv::Point>& contour);

    const cv::Mat& inputImage;
    const std::vector<cv::Point>& contour;
};

// In order to provide custom criteria, applications can derive from this class.
// For samples, see ComponentFilterSamples.h and .cpp.
class ComponentFilterCriterion
{
public:
    virtual ~ComponentFilterCriterion() {}

    // Return true if the component shall be accepted.
    virtual bool operator()(ComponentFilterCriterionInput& input) const = 0;
};

// Helper struct to allow keeping the allocated memory instead of having to allocate it over and over again.
struct ComponentFilterTemp
{
    std::vector<std::vector<cv::Point>> contours;
};

void ComponentFilter(const cv::Mat& input, cv::Mat& output, const ComponentFilterCriterion& criterion, ComponentFilterTemp& temp = ComponentFilterTemp());

#endif // PUUKOT_COMPONENT_FILTER
