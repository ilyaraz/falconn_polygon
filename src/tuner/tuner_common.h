#ifndef _TUNER_COMMON_H_
#define _TUNER_COMMON_H_

#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace falconn {
namespace tuner {

using std::runtime_error;
using std::string;
using std::vector;

using Eigen::VectorXf;

void read_dataset(string file_name, vector<VectorXf> *dataset) {
    dataset->clear();
    uint32_t n, d;
    FILE *input = fopen(file_name.c_str(), "rb");
    if (fread(&n, sizeof(uint32_t), 1, input) != 1) {
        throw runtime_error("fread");
    }
    if (fread(&d, sizeof(uint32_t), 1, input) != 1) {
        throw runtime_error("fread");
    }
    for (uint32_t i = 0; i < n; ++i) {
        VectorXf row(d);
        if (fread(&row[0], sizeof(float), d, input) != d) {
            throw runtime_error("fread");
        }
        dataset->push_back(row);
    }
    fclose(input);
}

void write_dataset(string file_name, const vector<VectorXf> &dataset) {
    uint32_t n, d;
    n = dataset.size();
    d = dataset[0].size();
    for (uint32_t i = 0; i < n; ++i) {
        if (dataset[i].size() != d) {
            throw runtime_error("wrong dimension");
        }
    }
    FILE *output = fopen(file_name.c_str(), "wb");
    if (fwrite(&n, sizeof(uint32_t), 1, output) != 1) {
        throw runtime_error("fwrite");
    }
    if (fwrite(&d, sizeof(uint32_t), 1, output) != 1) {
        throw runtime_error("fwrite");
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (fwrite(&dataset[i][0], sizeof(float), d, output) != d) {
            throw runtime_error("fwrite");
        }
    }
    fclose(output);
}

void read_knn(string file_name, vector<vector<uint32_t>> *knn) {
    knn->clear();
    uint32_t n, d;
    FILE *input = fopen(file_name.c_str(), "rb");
    if (fread(&n, sizeof(uint32_t), 1, input) != 1) {
        throw runtime_error("fread");
    }
    if (fread(&d, sizeof(uint32_t), 1, input) != 1) {
        throw runtime_error("fread");
    }
    for (uint32_t i = 0; i < n; ++i) {
        vector<uint32_t> row(d);
        if (fread(&row[0], sizeof(uint32_t), d, input) != d) {
            throw runtime_error("fread");
        }
        knn->push_back(row);
    }
    fclose(input);
}

void write_knn(string file_name, const vector<vector<uint32_t>> &knn) {
    uint32_t n, d;
    n = knn.size();
    d = knn[0].size();
    for (uint32_t i = 0; i < n; ++i) {
        if (knn[i].size() != d) {
            throw runtime_error("wrong dimension");
        }
    }
    FILE *output = fopen(file_name.c_str(), "wb");
    if (fwrite(&n, sizeof(uint32_t), 1, output) != 1) {
        throw runtime_error("fwrite");
    }
    if (fwrite(&d, sizeof(uint32_t), 1, output) != 1) {
        throw runtime_error("fwrite");
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (fwrite(&knn[i][0], sizeof(uint32_t), d, output) != d) {
            throw runtime_error("fwrite");
        }
    }
    fclose(output);
}

}
}

#endif
