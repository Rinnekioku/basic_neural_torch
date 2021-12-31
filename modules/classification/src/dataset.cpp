#include "dataset.h"

namespace classification {

namespace fs = std::filesystem;

CustomDataset::CustomDataset(std::string dataset_root_dir)
    : dataset_(ReadDataset(dataset_root_dir)) {}

torch::data::Example<> CustomDataset::get(size_t index) {
  std::string file_location = std::get<0>(dataset_[index]);
  int64_t label = std::get<1>(dataset_[index]);
  cv::Mat img = cv::imread(file_location);
  torch::Tensor img_tensor =
      torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).clone();
  img_tensor = img_tensor.permute({2, 0, 1});  // convert to CxHxW

  torch::Tensor label_tensor = torch::full({1}, label);

  return {img_tensor, label_tensor};
}

torch::optional<size_t> CustomDataset::size() const { return dataset_.size(); }

std::vector<std::tuple<std::string, int64_t>> CustomDataset::ReadDataset(
    std::string dataset_location) {
  std::vector<std::tuple<std::string, int64_t>> dataset;

  for (int64_t current_label = 0;
       auto const& images_group :
       fs::directory_iterator(fs::path(dataset_location))) {
    for (auto const& image : fs::directory_iterator(images_group)) {
      dataset.push_back({image.path().string(), current_label});
    }

    current_label++;
  }

  return dataset;
}

}  // namespace classification
