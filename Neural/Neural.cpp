#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <algorithm>  
#include <numeric>   

const int IMAGE_ROWS = 28;
const int IMAGE_COLS = 28;
const int IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS;

// Function to reverse endianess
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to read the MNIST images
void readMNISTImages(const std::string& filename, std::vector<std::vector<float>>& images, int num_images) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&n_images, sizeof(n_images));
        n_images = reverseInt(n_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        for (int i = 0; i < num_images; ++i) {
            std::vector<float> image(IMAGE_SIZE);
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                image[j] = static_cast<float>(temp) / 255.0f; // Normalize
            }
            images.push_back(image);
        }
    }
}

// Function to read the MNIST labels
void readMNISTLabels(const std::string& filename, std::vector<int>& labels, int num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0, n_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&n_labels, sizeof(n_labels));
        n_labels = reverseInt(n_labels);

        for (int i = 0; i < num_labels; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels.push_back(static_cast<int>(temp));
        }
    }
}

// Function to shuffle data
template <typename T>
void shuffleData(std::vector<std::vector<float>>& images, std::vector<T>& labels) {
    srand(static_cast<unsigned>(time(0)));
    for (size_t i = 0; i < images.size(); ++i) {
        int j = rand() % images.size();
        std::swap(images[i], images[j]);
        std::swap(labels[i], labels[j]);
    }
}

Eigen::MatrixXd oneHotEncode(const std::vector<int>& labels, int numClasses) {
    int numSamples = labels.size();
    Eigen::MatrixXd oneHot = Eigen::MatrixXd::Zero(numClasses, numSamples);

    for (int i = 0; i < numSamples; ++i) {
        int label = labels[i];
        if (label >= 0 && label < numClasses) {
            oneHot(label, i) = 1.0;
        }
    }

    return oneHot;
}

Eigen::VectorXd softmax(const Eigen::VectorXd& Z) {
    Eigen::VectorXd expZ = Z.array().exp();
    double sumExpZ = expZ.sum();
    return expZ / sumExpZ;
}

Eigen::ArrayXXd softmaxDerivative(const Eigen::ArrayXXd& Z) {
    Eigen::ArrayXXd S = softmax(Z);
    return S * (1.0 - S);
}

Eigen::ArrayXXd sigmoid(const Eigen::ArrayXXd& Z) {
    return 1.0 / (1.0 + (-Z).exp());
}

Eigen::ArrayXXd sigmoidDerivative(const Eigen::ArrayXXd& Z) {
    Eigen::ArrayXXd sig = sigmoid(Z);
    return sig * (1.0 - sig);
}

Eigen::ArrayXXd relu(const Eigen::ArrayXXd& Z) {
    return Z.max(0.0);
}

Eigen::ArrayXXd reluDerivative(const Eigen::ArrayXXd& Z) {
    return (Z > 0.0).cast<double>();
}

void forwardPass(const Eigen::MatrixXd& input,
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& bias,
    const Eigen::MatrixXd& weightsOutput,
    const Eigen::MatrixXd& biasOutput,
    Eigen::VectorXd& Z1,
    Eigen::VectorXd& A1,
    Eigen::VectorXd& Z2,
    Eigen::VectorXd& A2) {

    Z1 = weights * input + bias; 
    A1 = relu(Z1);

    Z2 = weightsOutput * A1 + biasOutput; 
    A2 = softmax(Z2); 
}

void backwardPass(const Eigen::VectorXd& input,
    const Eigen::VectorXd& A1,
    const Eigen::VectorXd& A2,
    const Eigen::MatrixXd& weights1,
    const Eigen::MatrixXd& weights2,
    const Eigen::VectorXd& oneHotY,
    Eigen::VectorXd& dZ1,
    Eigen::VectorXd& dZ2,
    Eigen::MatrixXd& dW1,
    Eigen::VectorXd& db1,
    Eigen::MatrixXd& dW2,
    Eigen::VectorXd& db2) {

    
    dZ2 = A2 - oneHotY; 

    dW2 = dZ2 * A1.transpose(); 
    db2 = dZ2;                  
    
    Eigen::VectorXd reluDerivA1 = reluDerivative(A1); 
    dZ1 = (weights2.transpose() * dZ2).array() * reluDerivA1.array();


    dW1 = dZ1 * input.transpose();
    db1 = dZ1;            
}

void evaluateModel(const std::vector<std::vector<float>>& test_images,
    const std::vector<int>& test_labels,
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& weightsOutput,
    const Eigen::VectorXd& bias,
    const Eigen::VectorXd& biasOutput) {

    int correct_predictions = 0;
    int total_samples = test_labels.size();

    for (int i = 0; i < total_samples; ++i) {
        Eigen::VectorXd input(784);
        for (int k = 0; k < 784; ++k) {
            input(k) = test_images[i][k];
        }

        Eigen::VectorXd Z1, A1, Z2, A2;
        forwardPass(input, weights, bias, weightsOutput, biasOutput, Z1, A1, Z2, A2);

        A2 = softmax(Z2);

        int predicted_label;
        A2.maxCoeff(&predicted_label); 

        if (predicted_label == test_labels[i]) {
            ++correct_predictions;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / total_samples * 100.0;
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
}


int main() {

    std::string train_image_file = ""; // Path to dataset 
    std::string train_label_file = "";
    std::string test_image_file = ""; 
    std::string test_label_file = "";

    int num_train = 60000;  
    int num_test = 10000;

    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;

    std::vector<std::vector<float>> test_images;
    std::vector<int> test_labels;

    // Read training and testing datasets
    readMNISTImages(train_image_file, train_images, num_train);
    readMNISTLabels(train_label_file, train_labels, num_train);

    readMNISTImages(test_image_file, test_images, num_test);
    readMNISTLabels(test_label_file, test_labels, num_test);

    // Shuffle the training data
    shuffleData(train_images, train_labels);

    // Split into training and development sets
    int dev_size = 1000;
    std::vector<std::vector<float>> X_dev(train_images.begin(), train_images.begin() + dev_size);
    std::vector<int> Y_dev(train_labels.begin(), train_labels.begin() + dev_size);

    std::vector<std::vector<float>> X_train(train_images.begin() + dev_size, train_images.end());
    std::vector<int> Y_train(train_labels.begin() + dev_size, train_labels.end());

    // Data is now loaded, shuffled, normalized, and split into dev and train sets
    std::cout << "Training set size: " << X_train.size() << " images" << std::endl;
    std::cout << "Development set size: " << X_dev.size() << " images" << std::endl;

    readMNISTLabels("path/to/train-labels.idx1-ubyte", train_labels, 60000);

    Eigen::MatrixXd oneHotLabels = oneHotEncode(train_labels, 10);

    std::cout << "First label: " << train_labels[0] << std::endl;
    std::cout << "One-hot encoding of the first label:\n" << oneHotLabels.col(0).transpose() << std::endl; 

    Eigen::VectorXd input(784); 
    input.setRandom();

    Eigen::MatrixXd weights(10, 784);
    weights.setRandom();  

    Eigen::VectorXd bias(10);
    bias.setRandom();

    Eigen::VectorXd biasOutput(10);
    biasOutput.setRandom();

    Eigen::MatrixXd weightsOutput(10, 10);
    weightsOutput.setRandom();

    Eigen::VectorXd Z1(10);
    Eigen::VectorXd Z2(10);
    Eigen::VectorXd A2(10);
    Eigen::VectorXd A1(10);

    Eigen::VectorXd dZ1(10), dZ2(10);
    Eigen::MatrixXd dW1(10, 784), dW2(10, 10);
    Eigen::VectorXd db1(10), db2(10);

    Eigen::MatrixXd oneHotY = oneHotEncode(train_labels, 10);

    int sayac = 0;
    int epochs = 10;       
    int batch_size = 32;    
    double learning_rate = 0.01; 
        // Train loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (int i = 0; i < num_train; i += batch_size) {
            
            for (int j = 0; j < batch_size; ++j) {
                if (i + j >= num_train) break;

                Eigen::VectorXd input(784);
                for (int k = 0; k < 784; ++k) {
                    input(k) = train_images[i + j][k];
                }
                sayac++;
                std::cout << sayac << std::endl;
                Eigen::VectorXd oneHotY(10);
                oneHotY.setZero();
                oneHotY(train_labels[i + j]) = 1.0;
            
                Eigen::VectorXd Z1(10), A1(10), Z2(10), A2(10);
                forwardPass(input, weights, bias, weightsOutput, biasOutput, Z1, A1, Z2, A2);
                Eigen::VectorXd dZ1(10), dZ2(10);
                Eigen::MatrixXd dW1(10, 784), dW2(10, 10);
                Eigen::VectorXd db1(10), db2(10);
                backwardPass(input, A1, A2, weights, weightsOutput, oneHotY, dZ1, dZ2, dW1, db1, dW2, db2);

               
                weights -= learning_rate * dW1; 
                bias -= learning_rate * db1;
                weightsOutput -= learning_rate * dW2;
                biasOutput -= learning_rate * db2;

                double loss = (A2 - oneHotY).array().square().sum() / 2.0;
                epoch_loss += loss;

            }
            
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss / num_train << std::endl;
        evaluateModel(test_images, test_labels, weights, weightsOutput, bias, biasOutput);
    }
    std::cout << "Training Completed" << std::endl;

    return 0;
}
