#include "picotest/picotest.h"
#include "../../PyrMatchTemplate.h"
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

cv::Mat GeneratePyrTemplateMatchInputImage()
{
    cv::Mat image(6000, 8000, CV_8UC1);
    cv::randn(image, 64, 32);

    cv::Mat circle(6000, 8000, CV_8UC1);
    circle = 0;
    cv::circle(circle, cv::Point(4000, 3000), 2000, 128, cv::FILLED);

    image += circle;

    return image;
}

TEST_F(TestPyrMatchTemplate, PyramidBasedTemplateMatchingWorks)
{
    cv::Mat inputImage = GeneratePyrTemplateMatchInputImage();
    cv::Mat templ = inputImage(cv::Rect(5000, 4000, 1000, 1000));

}

TEST_F(TestPyrMatchTemplate, PyramidBasedTemplateMatchingIsFast)
{
    cv::Mat inputImage = GeneratePyrTemplateMatchInputImage();
    cv::Mat templ = inputImage(cv::Rect(5000, 4000, 1000, 1000));

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

    EXPECT_LT(duration2_ms, duration1_ms * 0.8);
}
