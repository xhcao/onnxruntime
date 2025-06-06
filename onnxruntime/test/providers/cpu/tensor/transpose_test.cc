// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

template <typename T>
class TransposeOpTest : public ::testing::Test {
};

using TransposeOpTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(TransposeOpTest, TransposeOpTestTypes);

TEST(TransposeOpTest, IsTransposeReshapeTest) {
  std::vector<int64_t> input_dims{1, 2, 3, 4, 1};
  std::vector<size_t> perm{0, 1, 2, 3, 4};
  ASSERT_TRUE(IsTransposeReshape(perm, input_dims));
  perm = std::vector<size_t>{1, 2, 3, 0, 4};
  ASSERT_TRUE(IsTransposeReshape(perm, input_dims));
  perm = std::vector<size_t>{4, 1, 0, 2, 3};
  ASSERT_TRUE(IsTransposeReshape(perm, input_dims));
  perm = std::vector<size_t>{4, 1, 0, 3, 2};
  ASSERT_FALSE(IsTransposeReshape(perm, input_dims));
}

// Negative test, making sure it fails
TEST(TransposeOpTest, PermRankDoesNotMatchTensorRank) {
  const std::vector<float> input_vals(1 * 2 * 3 * 4, 0.f);
  const std::vector<int64_t> perm{0, 2, 1};

  OpTester test("Transpose");
  test.AddAttribute("perm", perm);
  test.AddInput<float>("X", {1, 2, 3, 4}, input_vals);
  // Output is not very relevant
  test.AddOutput<float>("Y", {1, 3, 2, 4}, input_vals);
  // This failure comes from shape inference, because in this case it knows the input dims.
  // But in the real world, the model can supply different input dims at runtime.
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Node:node1 Output:Y [ShapeInferenceError] Mismatch between number of inferred and declared dimensions. inferred=3 declared=4");
}

// Some of the tests can't run on TensorrtExecutionProvider because of errors.
// Those tests will fallback to other EPs.

template <class T>
void TransposeTest(const std::vector<int64_t>& input_shape,
                   const std::vector<T>& input_vals,
                   const std::vector<int64_t>* p_perm,
                   const std::vector<int64_t>& expected_shape,
                   const std::vector<T>& expected_vals,
                   const std::unordered_set<std::string>& excluded_provider_types = {},
                   const std::vector<int>& opsets = {7}) {
  for (auto opset : opsets) {
    OpTester test("Transpose", opset);
    if (nullptr != p_perm)
      test.AddAttribute("perm", *p_perm);
    test.AddInput<T>("X", input_shape, input_vals);
    test.AddOutput<T>("Y", expected_shape, expected_vals);

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_provider_types);
  }
}

// Test 2 dimensional transpose, with no permutation attribute specified
TYPED_TEST(TransposeOpTest, TwoDimNoAttr) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<TypeParam> input_vals = GetTypedArray<TypeParam>({1.0f, 2.0f, 3.0f,
                                                                4.0f, 5.0f, 6.0f});

  std::vector<int64_t> expected_shape({3, 2});
  std::vector<TypeParam> expected_vals = GetTypedArray<TypeParam>({1.0f, 4.0f,
                                                                   2.0f, 5.0f,
                                                                   3.0f, 6.0f});

  TransposeTest(input_shape, input_vals, nullptr, expected_shape, expected_vals,
                {kTensorrtExecutionProvider}, {7, 21});  // TensorRT: SegFault error
}

TEST(TransposeOpTest, TwoDimNoAttrStr) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6"};

  std::vector<int64_t> expected_shape({3, 2});
  std::vector<std::string> expected_vals = {
      "1", "4",
      "2", "5",
      "3", "6"};

  TransposeTest(input_shape, input_vals, nullptr, expected_shape, expected_vals, {}, {7, 21});
}

// Test 2 dimensional transpose, with permutation attribute specified
TEST(TransposeOpTest, TwoDim) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<float> expected_vals = {1.0f, 4.0f,
                                      2.0f, 5.0f,
                                      3.0f, 6.0f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {}, {7, 21});
}

// Test Int4 transpose with odd inner dimension.
TEST(TransposeOpTest, TwoDim_Odd_Int4) {
  constexpr int8_t unused_val = 0;
  std::vector<int64_t> input_shape({5, 3});
  std::vector<Int4x2> input_vals = {Int4x2(1, 2), Int4x2(3, 4), Int4x2(5, 6), Int4x2(7, 8),
                                    Int4x2(9, 10), Int4x2(11, 12), Int4x2(13, 14), Int4x2(15, unused_val)};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 5});
  std::vector<Int4x2> expected_vals = {Int4x2(1, 4), Int4x2(7, 10), Int4x2(13, 2), Int4x2(5, 8),
                                       Int4x2(11, 14), Int4x2(3, 6), Int4x2(9, 12), Int4x2(15, unused_val)};

  OpTester test("Transpose", 21);
  test.AddAttribute("perm", perm);
  test.AddInput<Int4x2>("X", input_shape, input_vals);
  test.AddOutput<Int4x2>("Y", expected_shape, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test UInt4 transpose with odd inner dimension.
TEST(TransposeOpTest, TwoDim_Odd_UInt4) {
  constexpr int8_t unused_val = 0;
  std::vector<int64_t> input_shape({5, 3});
  std::vector<UInt4x2> input_vals = {UInt4x2(1, 2), UInt4x2(3, 4), UInt4x2(5, 6), UInt4x2(7, 8),
                                     UInt4x2(9, 10), UInt4x2(11, 12), UInt4x2(13, 14), UInt4x2(15, unused_val)};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 5});
  std::vector<UInt4x2> expected_vals = {UInt4x2(1, 4), UInt4x2(7, 10), UInt4x2(13, 2), UInt4x2(5, 8),
                                        UInt4x2(11, 14), UInt4x2(3, 6), UInt4x2(9, 12), UInt4x2(15, unused_val)};

  OpTester test("Transpose", 21);
  test.AddAttribute("perm", perm);
  test.AddInput<UInt4x2>("X", input_shape, input_vals);
  test.AddOutput<UInt4x2>("Y", expected_shape, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TransposeOpTest, TwoDim_double) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<double> input_vals = {1.0, 2.0, 3.0,
                                    4.0, 5.0, 6.0};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<double> expected_vals = {1.0, 4.0,
                                       2.0, 5.0,
                                       3.0, 6.0};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDim_int32) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int32_t> input_vals = {1, 2, 3,
                                     4, 5, 6};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<int32_t> expected_vals = {1, 4,
                                        2, 5,
                                        3, 6};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {}, {7, 21});
}

TEST(TransposeOpTest, TwoDim_int16) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int16_t> input_vals = {
      1, 2, 3,
      4, 5, 6};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<int16_t> expected_vals = {
      1, 4,
      2, 5,
      3, 6};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {kOpenVINOExecutionProvider}, {7, 21});
}

TEST(TransposeOpTest, TwoDim_mlfloat16) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<MLFloat16> input_vals;
  for (uint16_t i = 0; i < 6; ++i)
    input_vals.push_back(MLFloat16::FromBits(static_cast<uint16_t>(i)));

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<MLFloat16> expected_vals =
      {MLFloat16::FromBits(static_cast<uint16_t>(1)), MLFloat16::FromBits(static_cast<uint16_t>(4)),
       MLFloat16::FromBits(static_cast<uint16_t>(2)), MLFloat16::FromBits(static_cast<uint16_t>(5)),
       MLFloat16::FromBits(static_cast<uint16_t>(3)), MLFloat16::FromBits(static_cast<uint16_t>(6))};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {kTensorrtExecutionProvider}, {7, 21});
}

#if defined(USE_DNNL)
TEST(TransposeOpTest, TwoDim_opset13_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Transpose", 13);
  test.AddAttribute("perm", std::vector<int64_t>{1, 0});
  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddOutput<BFloat16>("transposed", {3, 2}, MakeBFloat16({1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(TransposeOpTest, TwoDimNoAttr_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Transpose", 13);
  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddOutput<BFloat16>("transposed", {3, 2}, MakeBFloat16({1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(TransposeOpTest, Transpose021_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  std::vector<int64_t> perm = {0, 2, 1};
  std::vector<int64_t> expected_shape({4, 3, 2});
  std::vector<float> expected_vals = {
      1.0f, 4.0f,
      2.0f, 5.0f,
      3.0f, 6.0f,

      1.1f, 4.1f,
      2.1f, 5.1f,
      3.1f, 6.1f,

      1.2f, 4.2f,
      2.2f, 5.2f,
      3.2f, 6.2f,

      1.3f, 4.3f,
      2.3f, 5.3f,
      3.3f, 6.3f};
  OpTester test("Transpose", 13);
  test.AddAttribute("perm", perm);
  test.AddInput<BFloat16>("data", input_shape, FloatsToBFloat16s(input_vals));
  test.AddOutput<BFloat16>("transposed", expected_shape, FloatsToBFloat16s(expected_vals));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(TransposeOpTest, TwoDim_int8) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int8_t> input_vals = {1, 2, 3,
                                    4, 5, 6};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<int8_t> expected_vals = {1, 4,
                                       2, 5,
                                       3, 6};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {kTensorrtExecutionProvider}, {7, 21});
}

TEST(TransposeOpTest, TwoDimStr) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6"};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::vector<std::string> expected_vals = {
      "1", "4",
      "2", "5",
      "3", "6"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {}, {7, 21});
}

// Test 3 dimensional transpose, with permutation attribute specified
TEST(TransposeOpTest, Transpose021) {
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  std::vector<int64_t> perm = {0, 2, 1};
  std::vector<int64_t> expected_shape({4, 3, 2});
  std::vector<float> expected_vals = {
      1.0f, 4.0f,
      2.0f, 5.0f,
      3.0f, 6.0f,

      1.1f, 4.1f,
      2.1f, 5.1f,
      3.1f, 6.1f,

      1.2f, 4.2f,
      2.2f, 5.2f,
      3.2f, 6.2f,

      1.3f, 4.3f,
      2.3f, 5.3f,
      3.3f, 6.3f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kTensorrtExecutionProvider}, {7, 21});  // TensorRT: illegal error
}

TEST(TransposeOpTest, Transpose120) {
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  std::vector<int64_t> perm = {1, 2, 0};
  std::vector<int64_t> expected_shape({2, 3, 4});
  std::vector<float> expected_vals = {
      1.0f, 1.1f, 1.2f, 1.3f,
      2.0f, 2.1f, 2.2f, 2.3f,
      3.0f, 3.1f, 3.2f, 3.3f,

      4.0f, 4.1f, 4.2f, 4.3f,
      5.0f, 5.1f, 5.2f, 5.3f,
      6.0f, 6.1f, 6.2f, 6.3f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kTensorrtExecutionProvider}, {7, 21});  // TensorRT: illegal error
}

// test when the suffix size is > 1 (last dimension is not moved)
TEST(TransposeOpTest, Transpose102) {
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  std::vector<int64_t> perm = {1, 0, 2};
  std::vector<int64_t> expected_shape({2, 4, 3});
  std::vector<float> expected_vals = {
      1.0f, 2.0f, 3.0f,
      1.1f, 2.1f, 3.1f,
      1.2f, 2.2f, 3.2f,
      1.3f, 2.3f, 3.3f,

      4.0f, 5.0f, 6.0f,
      4.1f, 5.1f, 6.1f,
      4.2f, 5.2f, 6.2f,
      4.3f, 5.3f, 6.3f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kTensorrtExecutionProvider}, {7, 21});  // TensorRT: illegal error
}

TEST(TransposeOpTest, TransposeReshape) {
  std::vector<int64_t> input_shape({1, 4, 2, 1, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  std::vector<int64_t> perm = {1, 3, 2, 4, 0};
  std::vector<int64_t> expected_shape({4, 1, 2, 3, 1});
  std::vector<float> expected_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kTensorrtExecutionProvider}, {7, 21});  // TensorRT: illegal error
}

TEST(TransposeOpTest, ThreeDimStr) {
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6",

      "1", "2", "3",
      "4", "5", "6",

      "1", "2", "3",
      "4", "5", "6",

      "1", "2", "3",
      "4", "5", "6"};

  std::vector<int64_t> perm = {0, 2, 1};
  std::vector<int64_t> expected_shape({4, 3, 2});
  std::vector<std::string> expected_vals = {
      "1", "4",
      "2", "5",
      "3", "6",

      "1", "4",
      "2", "5",
      "3", "6",

      "1", "4",
      "2", "5",
      "3", "6",

      "1", "4",
      "2", "5",
      "3", "6"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {}, {7, 21});
}

TEST(TransposeOpTest, SixDim) {
  // CoreML has a 5D limit. With the CoreML EP enabled, this should fall back to the CPU EP.
  const auto input_shape = std::vector<int64_t>{2, 2, 2, 2, 2, 2};
  const auto input_vals = []() {
    std::vector<float> v(64);
    std::iota(v.begin(), v.end(), 0.0f);
    return v;
  }();

  const auto perm = std::vector<int64_t>{1, 0, 2, 3, 4, 5};

  const auto expected_shape = input_shape;  // all dimension values are the same
  const auto expected_vals = []() {
    std::vector<float> v(64);
    std::iota(v.begin() + 0, v.begin() + 16, 0.f);
    std::iota(v.begin() + 16, v.begin() + 32, 32.0f);
    std::iota(v.begin() + 32, v.begin() + 48, 16.0f);
    std::iota(v.begin() + 48, v.begin() + 64, 48.0f);
    return v;
  }();

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kQnnExecutionProvider}, {7, 21});  // Error: Failed to finalize QNN graph.
}

template <typename T>
static void NumericNCHW2NHWC() {
  std::vector<int64_t> input_shape({1, 3, 2, 2});
  std::vector<T> input_vals = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12};

  std::vector<int64_t> perm = {0, 2, 3, 1};
  std::vector<int64_t> expected_shape({1, 2, 2, 3});
  std::vector<T> expected_vals = {
      1, 5, 9,
      2, 6, 10,
      3, 7, 11,
      4, 8, 12};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}
TEST(TransposeOpTest, NCHW2NHWC) {
  NumericNCHW2NHWC<int8_t>();
  NumericNCHW2NHWC<int16_t>();
  NumericNCHW2NHWC<uint32_t>();
  NumericNCHW2NHWC<uint64_t>();
}

TEST(TransposeOpTest, NCHW2NHWCStr) {
  std::vector<int64_t> input_shape({1, 3, 2, 2});
  std::vector<std::string> input_vals = {
      "1", "2", "3", "4",
      "5", "6", "7", "8",
      "9", "10", "11", "12"};

  std::vector<int64_t> perm = {0, 2, 3, 1};
  std::vector<int64_t> expected_shape({1, 2, 2, 3});
  std::vector<std::string> expected_vals = {
      "1", "5", "9",
      "2", "6", "10",
      "3", "7", "11",
      "4", "8", "12"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {kTensorrtExecutionProvider}, {7, 21});
}

template <typename T>
static void NumericNHWC2NCHW() {
  std::vector<int64_t> input_shape({2, 2, 2, 2});
  std::vector<T> input_vals = {
      1, 2,
      3, 4,

      5, 6,
      7, 8,

      9, 10,
      11, 12,

      13, 14,
      15, 16};

  std::vector<int64_t> perm = {0, 3, 1, 2};
  std::vector<int64_t> expected_shape({2, 2, 2, 2});
  std::vector<T> expected_vals = {
      1, 3,
      5, 7,

      2, 4,
      6, 8,

      9, 11,
      13, 15,

      10, 12,
      14, 16};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals,
                {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(TransposeOpTest, NHWC2NCHW) {
  NumericNHWC2NCHW<uint8_t>();
  NumericNHWC2NCHW<int16_t>();
  NumericNHWC2NCHW<uint32_t>();
  NumericNHWC2NCHW<int64_t>();
}

TEST(TransposeOpTest, NHWC2NCHW_String) {
  std::vector<int64_t> input_shape({1, 2, 2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6",
      "7", "8", "9",
      "10", "11", "12"};

  std::vector<int64_t> perm = {0, 3, 1, 2};
  std::vector<int64_t> expected_shape({1, 3, 2, 2});
  std::vector<std::string> expected_vals = {
      "1", "4", "7", "10",
      "2", "5", "8", "11",
      "3", "6", "9", "12"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {kTensorrtExecutionProvider}, {7, 21});
}

// test to cover memcpy from single axis moving inwards path
TEST(TransposeOpTest, SingleAxisMovingInwardsBlockCopy) {
  std::vector<int64_t> input_shape({2, 2, 2, 2});
  std::vector<uint64_t> input_vals = {
      1, 2,
      3, 4,

      5, 6,
      7, 8,

      9, 10,
      11, 12,

      13, 14,
      15, 16};

  std::vector<int64_t> perm = {1, 2, 0, 3};
  std::vector<int64_t> expected_shape({2, 2, 2, 2});
  std::vector<uint64_t> expected_vals = {
      1, 2,
      9, 10,

      3, 4,
      11, 12,

      5, 6,
      13, 14,

      7, 8,
      15, 16};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, {kTensorrtExecutionProvider});
}

TEST(TransposeOpTest, NDim) {
  std::vector<int64_t> input_shape({2, 2, 2, 2});
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f};

  std::vector<int64_t> perm = {1, 0, 2, 3};
  std::vector<float> expected_vals = {1.0f, 2.0f, 3.0f, 4.0f,
                                      9.0f, 10.0f, 11.0f, 12.0f,
                                      5.0f, 6.0f, 7.0f, 8.0f,
                                      13.0f, 14.0f, 15.0f, 16.0f};
  TransposeTest(input_shape, input_vals, &perm, input_shape, expected_vals);

  perm = {1, 0, 3, 2};
  std::vector<float> expected_vals2 = {1.0f, 3.0f, 2.0f, 4.0f,
                                       9.0f, 11.0f, 10.0f, 12.0f,
                                       5.0f, 7.0f, 6.0f, 8.0f,
                                       13.0f, 15.0f, 14.0f, 16.0f};
  TransposeTest(input_shape, input_vals, &perm, input_shape, expected_vals2);
}

TEST(TransposeOpTest, DoTransposeImpl) {
  std::vector<int64_t> input_shape({5, 2, 1, 3});
  std::vector<float> input_vals(30);
  for (auto it = input_vals.begin(); it != input_vals.end(); ++it) {
    *it = static_cast<float>(std::distance(input_vals.begin(), it));
  }
  std::vector<int64_t> perm = {2, 1, 0, 3};
  std::vector<int64_t> expected_shape({1, 2, 5, 3});
  std::vector<float> expected_vals = {0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f,
                                      12.0f, 13.0f, 14.0f, 18.0f, 19.0f, 20.0f,
                                      24.0f, 25.0f, 26.0f, 3.0f, 4.0f, 5.0f,
                                      9.0f, 10.0f, 11.0f, 15.0f, 16.0f, 17.0f,
                                      21.0f, 22.0f, 23.0f, 27.0f, 28.0f, 29.0f};
  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, DoTransposeImplString) {
  std::vector<int64_t> input_shape({5, 2, 1, 3});
  std::vector<std::string> input_vals(30);
  for (auto it = input_vals.begin(); it != input_vals.end(); ++it) {
    *it = std::string("n") + std::to_string(static_cast<int>(std::distance(input_vals.begin(), it)));
  }
  std::vector<int64_t> perm = {2, 1, 0, 3};
  std::vector<int64_t> expected_shape({1, 2, 5, 3});
  std::vector<std::string> expected_vals = {"n0", "n1", "n2", "n6", "n7", "n8",
                                            "n12", "n13", "n14", "n18", "n19", "n20",
                                            "n24", "n25", "n26", "n3", "n4", "n5",
                                            "n9", "n10", "n11", "n15", "n16", "n17",
                                            "n21", "n22", "n23", "n27", "n28", "n29"};
  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, DoTransposeEltWise) {
  // Configuration where DoTransposeEltWise is called.
  std::vector<int64_t> input_shape({2, 2, 2, 2});
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f};

  std::vector<int64_t> perm = {1, 0, 3, 2};
  std::vector<float> expected_vals2 = {1.0f, 3.0f, 2.0f, 4.0f,
                                       9.0f, 11.0f, 10.0f, 12.0f,
                                       5.0f, 7.0f, 6.0f, 8.0f,
                                       13.0f, 15.0f, 14.0f, 16.0f};
  TransposeTest(input_shape, input_vals, &perm, input_shape, expected_vals2);

  // Specific test which tests that function DoTransposeEltWise does not
  // copy values outside the target buffer.
  TensorShape tensor_shape(input_shape);
  std::vector<size_t> stride(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    size_t inpdim = perm[i];
    if (inpdim + 1 < input_shape.size())
      stride[i] = tensor_shape.SizeFromDimension(inpdim + 1);
    else
      stride[i] = 1;
  }

  std::vector<float> input_vals_end = {1.0f, 2.0f, 3.0f, 4.0f,
                                       5.0f, 6.0f, 7.0f, 8.0f,
                                       9.0f, 10.0f, 11.0f, 12.0f,
                                       13.0f, 14.0f, 15.0f, 16.0f,
                                       -1.0f, -1.0f};
  std::vector<float> target(input_vals_end.size(), 17.0f);

  std::vector<float> expected_vals3 = {1.0f, 3.0f, 2.0f, 4.0f,
                                       9.0f, 11.0f, 10.0f, 12.0f,
                                       5.0f, 7.0f, 6.0f, 8.0f,
                                       13.0f, 15.0f, 14.0f, 16.0f,
                                       17.0f, 17.0f};

  ASSERT_STATUS_OK(DoTransposeEltWise(input_shape.size(), input_shape, 16,
                                      stride, (uint8_t*)input_vals_end.data(), (uint8_t*)target.data(),
                                      sizeof(float)));
  for (size_t i = 0; i < input_vals_end.size(); ++i) {
    ASSERT_TRUE(target[i] == expected_vals3[i]);
  }
}

#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
static void TestTranspose(
    const std::vector<int64_t>& perm,
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& y_dims,
    double error_tolerance = 1e-4) {
  CompareOpTester test{"Transpose"};

  RandomValueGenerator random{};
  const auto X_data = random.Uniform<float>(x_dims, -10.0, 10.0);
  const auto Y_data = FillZeros<float>(y_dims);

  test.AddAttribute("perm", perm);
  test.AddInput("X", x_dims, X_data);
  test.AddOutput("Y", y_dims, Y_data);

  test.CompareWithCPU(kGpuExecutionProvider, error_tolerance);
}

TEST(TransposeOpTest, Transpose0213) {  // Will trigger Transpose4DParallelizeMultipleElementsPerThreadInInnermostDim()
  const std::vector<int64_t> X_dims{64, 128, 16, 64};
  const std::vector<int64_t> perm{0, 2, 1, 3};
  const std::vector<int64_t> Y_dims{64, 16, 128, 64};
  TestTranspose(perm, X_dims, Y_dims);
}

TEST(TransposeOpTest, Transpose0213_V2) {  // Will trigger Transpose4DParallelizeOneElementPerThread()
  const std::vector<int64_t> X_dims{64, 128, 64, 2};
  const std::vector<int64_t> perm{0, 2, 1, 3};
  const std::vector<int64_t> Y_dims{64, 64, 128, 2};
  TestTranspose(perm, X_dims, Y_dims);
}

TEST(TransposeOpTest, Transpose3DImpl) {
  // Flattening dims 2 and 3 into one dim.
  {
    const std::vector<int64_t> X_dims{64, 128, 16, 64};
    const std::vector<int64_t> perm{0, 2, 3, 1};
    const std::vector<int64_t> Y_dims{64, 16, 64, 128};
    TestTranspose(perm, X_dims, Y_dims);
  }

  // Flattening dims 1 and 2 into one dim.
  {
    const std::vector<int64_t> X_dims{64, 16, 64, 128};
    const std::vector<int64_t> perm{0, 3, 1, 2};
    const std::vector<int64_t> Y_dims{64, 128, 16, 64};
    TestTranspose(perm, X_dims, Y_dims);
  }

  // dim-1 or dim-2 is not power of 2.
  {
    const std::vector<int64_t> X_dims{64, 12, 128};
    const std::vector<int64_t> perm{0, 2, 1};
    const std::vector<int64_t> Y_dims{64, 128, 12};
    TestTranspose(perm, X_dims, Y_dims);
  }

  {
    const std::vector<int64_t> X_dims{64, 99, 24};
    const std::vector<int64_t> perm{0, 2, 1};
    const std::vector<int64_t> Y_dims{64, 24, 99};
    TestTranspose(perm, X_dims, Y_dims);
  }
}

static void TestTransposeMLFloat16(
    const std::vector<int64_t>& perm,
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& y_dims,
    double error_tolerance = 1e-4) {
  CompareOpTester test{"Transpose"};

  RandomValueGenerator random{};
  const auto X_data = random.Uniform<float>(x_dims, -10.0, 10.0);
  std::vector<MLFloat16> X_data_f16 = FloatsToMLFloat16s(X_data);
  const auto Y_data = FillZeros<float>(y_dims);
  std::vector<MLFloat16> Y_data_f16 = FloatsToMLFloat16s(Y_data);

  test.AddAttribute("perm", perm);
  test.AddInput("X", x_dims, X_data_f16);
  test.AddOutput("Y", y_dims, Y_data_f16);

  test.CompareWithCPU(kGpuExecutionProvider, error_tolerance);
}

TEST(TransposeOpTest, TransposeBigMLFloat16) {  // Exercises CanUse_cublasTransposeHelper_MLFloat16 (cuda 65535 grid dimension limit)
  const std::vector<int64_t> X_dims{1, 1449, 1449, 3};
  const std::vector<int64_t> perm{0, 3, 1, 2};
  const std::vector<int64_t> Y_dims{1, 1449, 1449, 3};
  TestTransposeMLFloat16(perm, X_dims, Y_dims);
}
#endif  // defined(USE_CUDA) || defined(USE_ROCM)

}  // namespace test
}  // namespace onnxruntime
