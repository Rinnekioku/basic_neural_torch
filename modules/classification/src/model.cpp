#include "model.h"

namespace classification {

ConvNetImpl::ConvNetImpl(int64_t channels, int64_t height, int64_t width)
    : conv1(torch::nn::Conv2dOptions(3, 32, 4).stride(2)),
      conv2(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
      conv3(torch::nn::Conv2dOptions(64, 128, 4).stride(2)),
      flatten_length(GetFlattenLength(channels, height, width)),
      lin1(flatten_length, 512),
      lin2(512, 256),
      lin3(256, 128),
      lin4(128, 64),
      lin5(64, 9) {
  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("conv3", conv3);
  register_module("lin1", lin1);
  register_module("lin2", lin2);
  register_module("lin3", lin3);
  register_module("lin4", lin4);
  register_module("lin5", lin5);
};

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
  x = torch::relu(torch::max_pool2d(conv1(x), 2));
  x = torch::relu(torch::max_pool2d(conv2(x), 2));
  x = torch::relu(torch::max_pool2d(conv3(x), 2));

  // Flatten.
  x = x.view({-1, flatten_length});

  x = torch::relu(lin1(x));
  x = torch::relu(lin2(x));
  x = torch::relu(lin3(x));
  x = torch::relu(lin4(x));
  x = torch::log_softmax(lin5(x), 1);

  return x;
};

int64_t ConvNetImpl::GetFlattenLength(int64_t channels, int64_t height, int64_t width) {
  torch::Tensor x = torch::zeros({1, channels, height, width});
  x = torch::max_pool2d(conv1(x), 2);
  x = torch::max_pool2d(conv2(x), 2);
  x = torch::max_pool2d(conv3(x), 2);

  return x.numel();
}
}
