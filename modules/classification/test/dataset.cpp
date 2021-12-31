#include "dataset.h"

#include <gtest/gtest.h>

#include "classification.h"

namespace cls = classification;

TEST(classification_module, dataset) {
  cls::CustomDataset dataset("/home/rinnekioku/Projects/debug_dataset");

  EXPECT_EQ(90, dataset.size());
}
