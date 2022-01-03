#include <gtest/gtest.h>

#include "classification.h"

namespace cls = classification;

TEST(classification_module, dataset) {
  cls::CustomDataset dataset("../../data/geometric-shapes/validate");

  EXPECT_EQ(90, dataset.size());
}
