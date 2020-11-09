#include "picotest/picotest.h"

#include "../../PyrMatchTemplate.h"
#include "../../MinMax.h"

#include <opencv2/imgproc/imgproc.hpp>

class TestPyrMatchTemplate : public ::testing::Test {
protected:
    TestPyrMatchTemplate() {}
    virtual ~TestPyrMatchTemplate() {} // You can do clean-up work that doesn't throw exceptions here.

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    virtual void SetUp() {} // Code here will be called immediately after the constructor (right before each test).
    virtual void TearDown() {} // Code here will be called immediately after each test (right before the destructor).
};

cv::Mat GeneratePyrTemplateMatchInputImage(int scaler)
{
    cv::Mat image(600 * scaler, 800 * scaler, CV_8UC1);
    cv::randn(image, 64, 32);

    cv::Mat circle(image.size(), CV_8UC1);
    circle = 0;
    cv::circle(circle, cv::Point(400 * scaler, 300 * scaler), 200 * scaler, 128, cv::FILLED);

    image += circle;

    return image;
}

TEST_F(TestPyrMatchTemplate, PyramidBasedTemplateMatchingWorks)
{
    int scaler = 1;

    cv::Mat inputImage = GeneratePyrTemplateMatchInputImage(scaler);
    cv::Mat templ = inputImage(cv::Rect(500 * scaler, 400 * scaler, 100 * scaler, 100 * scaler));

    cv::Mat cvMatchTemplateResult, pyrMatchTemplateResult;

    {
        const int method = cv::TM_SQDIFF;

        cv::matchTemplate(inputImage, templ, cvMatchTemplateResult, method);
        PyrMatchTemplate(inputImage, templ, pyrMatchTemplateResult, method);

        const auto stdMinMax = FindMinMax(cvMatchTemplateResult);
        const auto pyrMinMax = FindMinMax(pyrMatchTemplateResult);

        EXPECT_EQ(stdMinMax.minLocation, pyrMinMax.minLocation);
        EXPECT_NEAR(stdMinMax.minValue, pyrMinMax.minValue, stdMinMax.maxValue * 1e-6);
    }

    {
        const int method = cv::TM_CCORR;

        cv::matchTemplate(inputImage, templ, cvMatchTemplateResult, method);
        PyrMatchTemplate(inputImage, templ, pyrMatchTemplateResult, method);

        const auto stdMinMax = FindMinMax(cvMatchTemplateResult);
        const auto pyrMinMax = FindMinMax(pyrMatchTemplateResult);

        EXPECT_EQ(stdMinMax.maxLocation, pyrMinMax.maxLocation);
        EXPECT_NEAR(stdMinMax.maxValue, pyrMinMax.maxValue, stdMinMax.maxValue * 1e-6);
    }
}

TEST_F(TestPyrMatchTemplate, PyramidBasedTemplateMatchingIsFast)
{
    int scaler = 10;

    cv::Mat inputImage = GeneratePyrTemplateMatchInputImage(scaler);
    cv::Mat templ = inputImage(cv::Rect(500 * scaler, 400 * scaler, 100 * scaler, 100 * scaler));

    cv::Mat cvMatchTemplateResult, pyrMatchTemplateResult;

    int method = cv::TM_SQDIFF;

    const auto t1 = std::chrono::steady_clock::now();

    cv::matchTemplate(inputImage, templ, cvMatchTemplateResult, method);

    const auto t2 = std::chrono::steady_clock::now();

    PyrMatchTemplate(inputImage, templ, pyrMatchTemplateResult, method);

    const auto t3 = std::chrono::steady_clock::now();

    const auto duration1 = t2 - t1;
    const auto duration2 = t3 - t2;

    const auto duration1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration1).count();
    const auto duration2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration2).count();

    // The pyramid-based method should be faster; otherwise it's just plain useless.
#ifdef _DEBUG
    EXPECT_LT(duration2_ms, duration1_ms * 0.8);
#else
    EXPECT_LT(duration2_ms, duration1_ms * 0.1);
#endif
}
