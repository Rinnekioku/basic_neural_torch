#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

class CustomDataset : public torch::data::Dataset<CustomDataset> {
 private:
  std::vector<std::tuple<std::string, int64_t>> dataset_;
  std::vector<std::tuple<std::string, int64_t>> ReadDataset(
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

 public:
  explicit CustomDataset(std::string dataset_root_dir)
      : dataset_(ReadDataset(dataset_root_dir)) {}

  torch::data::Example<> get(size_t index) override {
    std::string file_location = std::get<0>(dataset_[index]);
    int64_t label = std::get<1>(dataset_[index]);
    cv::Mat img = cv::imread(file_location);
    torch::Tensor img_tensor =
        torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte)
            .clone();
    img_tensor = img_tensor.permute({2, 0, 1});  // convert to CxHxW

    torch::Tensor label_tensor = torch::full({1}, label);

    return {img_tensor, label_tensor};
  }

  torch::optional<size_t> size() const override { return dataset_.size(); }
};

struct ConvNetImpl : public torch::nn::Module {
  ConvNetImpl(int64_t channels, int64_t height, int64_t width)
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

  torch::Tensor forward(torch::Tensor x) {
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

  int64_t GetFlattenLength(int64_t channels, int64_t height, int64_t width) {
    torch::Tensor x = torch::zeros({1, channels, height, width});
    x = torch::max_pool2d(conv1(x), 2);
    x = torch::max_pool2d(conv2(x), 2);
    x = torch::max_pool2d(conv3(x), 2);

    return x.numel();
  }

  torch::nn::Conv2d conv1, conv2, conv3;
  int64_t flatten_length;
  torch::nn::Linear lin1, lin2, lin3, lin4, lin5;
};

TORCH_MODULE(ConvNet);

void classify() {
  std::string image_path;

  std::cin >> image_path;

  cv::Mat image = cv::imread(image_path);

  torch::Tensor image_tensor = torch::from_blob(
      image.data, {1, image.rows, image.cols, 3}, torch::kByte);
  image_tensor = image_tensor.permute({0, 3, 1, 2});  // convert to CxHxW
  image_tensor = image_tensor.to(torch::kF32);

  ConvNet model(3, 200, 200);
  torch::load(model, "../../best_model.pt");

  torch::Tensor log_probability = model(image_tensor);
  torch::Tensor probability = torch::exp(log_probability);

  std::printf(
      "Probability of being\n\
    heptagon = %.2f percent\n\
    hexagon = %.2f percent\n\
    star = %.2f percent\n\
    square = %.2f percent\n\
    circle = %.2f percent\n\
    triangle = %.2f percent\n\
    nonagon = %.2f percent\n\
    octagon = %.2f percent\n\
    pentagon = %.2f percent\n",
      probability[0][0].item<float>() * 100.,
      probability[0][1].item<float>() * 100.,
      probability[0][2].item<float>() * 100.,
      probability[0][3].item<float>() * 100.,
      probability[0][4].item<float>() * 100.,
      probability[0][5].item<float>() * 100.,
      probability[0][6].item<float>() * 100.,
      probability[0][7].item<float>() * 100.,
      probability[0][8].item<float>() * 100.);
}

void train(std::string model_save_dir = "../..") {
  ConvNet model(3, 200, 200);

  std::string dataset_root_dir;

  std::cout << "Insert dataset root directory path" << std::endl;
  std::cin >> dataset_root_dir;

  auto data_set =
      CustomDataset(dataset_root_dir).map(torch::data::transforms::Stack<>());

  int64_t batch_size = 64;
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          data_set, batch_size);

  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(1e-3));

  int64_t n_epochs = 30;
  int64_t log_interval = 10;
  int dataset_size = data_set.size().value();

  float best_mse = std::numeric_limits<float>::max();

  for (int epoch = 1; epoch <= n_epochs; epoch++) {
    float mse = 0.f;
    int count = 0;

    for (size_t batch_idx = 0; auto& batch : *data_loader) {
      auto imgs = batch.data;
      auto labels = batch.target.squeeze();

      imgs = imgs.to(torch::kF32);
      labels = labels.to(torch::kInt64);

      optimizer.zero_grad();
      auto output = model(imgs);
      auto loss = torch::nll_loss(output, labels);

      loss.backward();
      optimizer.step();

      mse += loss.template item<float>();

      batch_idx++;
      if (batch_idx % log_interval == 0) {
        std::printf("\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f", epoch,
                    n_epochs, batch_idx * batch.data.size(0), dataset_size,
                    loss.template item<float>());
      }

      count++;
    }

    mse /= (float)count;
    printf(" Mean squared error: %f\n", mse);

    if (mse < best_mse) {
      torch::save(model, model_save_dir + "/best_model.pt");
      best_mse = mse;
    }
  }
}

int main() {
  classify();

  return 0;
}
