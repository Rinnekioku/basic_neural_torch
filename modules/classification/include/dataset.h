#ifndef CLASSIFICATION_DATASET_MODULE_H_
#define CLASSIFICATION_DATASET_MODULE_H_

#include <torch/torch.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

namespace classification {

class CustomDataset : public torch::data::Dataset<CustomDataset> {
 private:
  std::vector<std::tuple<std::string, int64_t>> dataset_;
  std::vector<std::tuple<std::string, int64_t>> ReadDataset(
      std::string dataset_location);

 public:
  explicit CustomDataset(std::string dataset_root_dir);

  torch::data::Example<> get(size_t index) override;

  torch::optional<size_t> size() const override;
};
}  // namespace classification

#endif
