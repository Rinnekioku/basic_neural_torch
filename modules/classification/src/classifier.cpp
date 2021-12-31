#include "classifier.h"

namespace classification {

using namespace std::string_literals;

template <typename ModuleType>
Classifier<ModuleType>::Classifier(ModuleType const model)
    : model_(model) {}

template <typename ModuleType>
template <typename... ModelParams>
Classifier<ModuleType>::Classifier(std::string model_path,
                                   ModelParams... model_params)
    : model_(ModuleType(model_params...)) {
  torch::load(model_, model_path);
}

template <typename ModuleType>
template <typename... ModelParams>
void Classifier<ModuleType>::LoadModel(ModuleType const model,
                                       ModelParams... model_params) {
  model_(model_params...);
}

template <typename ModuleType>
template <typename... ModelParams>
void Classifier<ModuleType>::LoadModel(std::string model_path,
                                       ModelParams... model_params) {
  model_(model_params...);
  torch::load(model_, model_path);
}

template <typename ModuleType>
int Classifier<ModuleType>::Classify(std::string image_path) {
  cv::Mat image = cv::imread(image_path);

  torch::Tensor image_tensor = torch::from_blob(
      image.data, {1, image.rows, image.cols, 3}, torch::kByte);
  image_tensor = image_tensor.permute({0, 3, 1, 2});  // convert to CxHxW
  image_tensor = image_tensor.to(torch::kF32);

  torch::Tensor log_probability = model_(image_tensor);
  torch::Tensor probability = torch::exp(log_probability);

  return torch::argmax(probability).item<int>();
}

template <typename ModuleType>
int Classifier<ModuleType>::Classify(cv::Mat const image) {
  torch::Tensor image_tensor = torch::from_blob(
      image.data, {1, image.rows, image.cols, 3}, torch::kByte);
  image_tensor = image_tensor.permute({0, 3, 1, 2});  // convert to CxHxW
  image_tensor = image_tensor.to(torch::kF32);

  torch::Tensor log_probability = model_(image_tensor);
  torch::Tensor probability = torch::exp(log_probability);

  return torch::argmax(probability).item<int>();
}

template <typename ModuleType>
template <typename DatasetType>
void Classifier<ModuleType>::Train(
    std::shared_ptr<torch::data::Dataset<DatasetType>> const dataset,
    std::string model_save_dir) {
  int64_t batch_size = 64;
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          dataset, batch_size);

  torch::optim::Adam optimizer(model_->parameters(),
                               torch::optim::AdamOptions(1e-3));

  int64_t n_epochs = 30;
  int64_t log_interval = 10;
  int dataset_size = dataset.size().value();

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
      torch::save(model_, model_save_dir + "/best_model.pt");
      best_mse = mse;
    }
  }
}

template class Classifier<classification::ConvNet>;

}  // namespace classification
