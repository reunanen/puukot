#include "ComponentFilter.h"

// This is a sample criterion: minimum contour length.
class MinimumContourLengthCriterion : public ComponentFilterCriterion
{
public:
    MinimumContourLengthCriterion(size_t minimumContourLength) : minimumContourLength(minimumContourLength) {}
    virtual ~MinimumContourLengthCriterion() {}
    virtual bool operator()(ComponentFilterCriterionInput& input) const;

private:
    const size_t minimumContourLength;
};

// Another sample criterion: minimum area.
class MinimumAreaCriterion : public ComponentFilterCriterion
{
public:
    MinimumAreaCriterion(int minimumArea) : minimumArea(minimumArea) {}
    virtual ~MinimumAreaCriterion() {}
    virtual bool operator()(ComponentFilterCriterionInput& input) const;

private:
    const int minimumArea;
    mutable cv::Mat temp;
};

// Yet another sample: minimum elongation.
class MinimumElongationCriterion : public ComponentFilterCriterion
{
public:
    MinimumElongationCriterion(double minimumElongation) : minimumElongation(minimumElongation) {}
    virtual ~MinimumElongationCriterion() {}
    virtual bool operator()(ComponentFilterCriterionInput& input) const;

private:
    const double minimumElongation;
    mutable cv::Mat temp;
};
