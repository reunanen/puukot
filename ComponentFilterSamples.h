#ifndef PUUKOT_COMPONENT_FILTER_SAMPLES
#define PUUKOT_COMPONENT_FILTER_SAMPLES

#include "ComponentFilter.h"

// This is a sample criterion: minimum contour length.
class MinimumContourLengthCriterion : public ComponentFilterCriterion
{
public:
    MinimumContourLengthCriterion(size_t minimumContourLength) : minimumContourLength(minimumContourLength) {}
    virtual ~MinimumContourLengthCriterion() {}
    virtual bool operator()(const ComponentFilterCriterionInput& input) const;

private:
    const size_t minimumContourLength;
};

// In practice, many criteria are based on a mask, so let's offer this class as a convenience.
class MaskBasedCriterion : public ComponentFilterCriterion
{
protected:
    const cv::Mat& GetMask(const ComponentFilterCriterionInput& input) const;

private:
    mutable cv::Mat mask;
};

// Another sample criterion: minimum area.
class MinimumAreaCriterion : public ComponentFilterCriterion
{
public:
    MinimumAreaCriterion(int minimumArea) : minimumArea(minimumArea) {}
    virtual ~MinimumAreaCriterion() {}
    virtual bool operator()(const ComponentFilterCriterionInput& input) const;

private:
    const int minimumArea;
};

// Yet another sample: minimum elongation.
class MinimumElongationCriterion : public MaskBasedCriterion
{
public:
    MinimumElongationCriterion(double minimumElongation) : minimumElongation(minimumElongation) {}
    virtual ~MinimumElongationCriterion() {}
    virtual bool operator()(const ComponentFilterCriterionInput& input) const;

private:
    const double minimumElongation;
};

#endif // PUUKOT_COMPONENT_FILTER_SAMPLES