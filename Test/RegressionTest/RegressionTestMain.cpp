#include "picotest/picotest.h"

int main(int argc, char **argv) {
    try {
        RUN_ALL_TESTS();
        return static_cast<int>(picotest::framework::Registry::getInstance().numFailed());
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 0xbadf00d;
    }
}
