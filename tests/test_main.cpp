#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <dexel/dexel.hpp>

TEST_CASE("Example test", "[dexel]") {
    dexel::MyTemplate<int> x;
    REQUIRE(true);  // 仮のテスト
}
