#ifndef CLASSIFICATION_CLASSIFIER_MODULE_H_
#define CLASSIFICATION_CLASSIFIER_MODULE_H_

#include <torch/torch.h>
#include "model.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>

namespace classification {

template <typename ModuleType>
class Classifier {
 public:
  Classifier() = delete;
  Classifier(ModuleType const model);
  template <typename... ModelParams>
  Classifier(std::string model_path, ModelParams... model_params);

  template <typename... ModelParams>
  void LoadModel(ModuleType const model,
                 ModelParams... model_params);
  template <typename... ModelParams>
  void LoadModel(std::string model_path, ModelParams... model_params);

  int Classify(std::string image_path);
  int Classify(cv::Mat const image);
  template <typename DatasetType>
  void Train(std::shared_ptr<torch::data::Dataset<DatasetType>> const dataset,
             std::string model_save_dir);

 private:
  ModuleType model_;
};

}  // namespace classification

#endif
