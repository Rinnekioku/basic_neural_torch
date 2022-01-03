#include <gtest/gtest.h>

#include "classification.h"

namespace cls = classification;

TEST(classification_module, classifier) {
  cls::ConvNet model(3, 200, 200);
  torch::load(model, "../../../best_model.pt");
  cls::Classifier<cls::ConvNet> classifier(model);

  std::string image_path =
      "../../data/geometric-shapes/validate/circle/"
      "Circle_0a0b51ca-2a86-11ea-8123-8363a7ec19e6.png";

  EXPECT_EQ(classifier.Classify(image_path), 4);
}
