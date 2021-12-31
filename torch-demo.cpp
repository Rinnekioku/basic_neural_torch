#include <classification.h>

#include <memory>
#include <vector>

using namespace std::string_literals;

namespace cls = classification;

int main() {
  cls::ConvNet model(3, 200, 200);
  torch::load(model, "../../best_model.pt");
  cls::Classifier<cls::ConvNet> classifier(model);

  const std::vector<std::string> shape_types = {
      "heptagon", "hexagon", "star",    "square",  "circle",
      "triangle", "nonagon", "octagon", "pentagon"};

  std::string image_path;

  std::cin >> image_path;

  std::cout << "Image type is: " << shape_types[classifier.Classify(image_path)]
            << std::endl;

  return 0;
}
