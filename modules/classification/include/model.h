#ifndef CLASSIFICATION_MODEL_MODULE_H_
#define CLASSIFICATION_MODEL_MODULE_H_

#include <torch/torch.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

namespace classification {
struct ConvNetImpl : public torch::nn::Module {
  ConvNetImpl(int64_t channels, int64_t height, int64_t width);

  torch::Tensor forward(torch::Tensor x);

  int64_t GetFlattenLength(int64_t channels, int64_t height, int64_t width);

 private:
  torch::nn::Conv2d conv1, conv2, conv3;
  int64_t flatten_length;
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5;
};

TORCH_MODULE(ConvNet);

}  // namespace classification

#endif
