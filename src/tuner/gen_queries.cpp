#include "tuner_common.h"

#include <Eigen/Dense>

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <cstdint>

using std::cout;
using std::endl;
using std::make_pair;
using std::mt19937_64;
using std::numeric_limits;
using std::pair;
using std::sort;
using std::uniform_int_distribution;
using std::vector;

using Eigen::VectorXf;

using falconn::tuner::read_dataset;
using falconn::tuner::write_dataset;
using falconn::tuner::write_knn;

int main() {
  const uint32_t q = 10000;
  const uint32_t k = 10;
  vector<VectorXf> dataset, queries;
  read_dataset("dataset.dat", &dataset);
  uint32_t n = dataset.size();
  uint32_t d = dataset[0].size();
  cout << n << " " << d << endl;
  cout << q << endl;
  mt19937_64 gen(4057218);
  for (uint32_t i = 0; i < q; ++i) {
    uniform_int_distribution<uint32_t> u(0, dataset.size() - 1);
    uint32_t ind = u(gen);
    queries.push_back(dataset[ind]);
    dataset[ind].swap(dataset.back());
    dataset.pop_back();
  }
  write_dataset("dataset_filtered.dat", dataset);
  write_dataset("queries.dat", queries);
  n = dataset.size();
  cout << n << " " << d << endl;
  vector<vector<uint32_t>> knn_output;
  for (uint32_t i = 0; i < q; ++i) {
    cout << i << ": ";
    vector<pair<float, uint32_t>> knn(
        k, make_pair(numeric_limits<float>::max(), n));
    pair<float, uint32_t> best(numeric_limits<float>::max(), 0);
    for (uint32_t j = 0; j < n; ++j) {
      float score = (queries[i] - dataset[j]).norm();
      if (score >= best.first) {
        continue;
      }
      knn[best.second] = make_pair(score, j);
      best = make_pair(-1.0, k);
      for (uint32_t kk = 0; kk < k; ++kk) {
        if (knn[kk].first > best.first) {
          best = make_pair(knn[kk].first, kk);
        }
      }
    }
    sort(knn.begin(), knn.end());
    vector<uint32_t> row(k);
    for (uint32_t kk = 0; kk < k; ++kk) {
      cout << "(" << knn[kk].first << ", " << knn[kk].second << ") ";
      row[kk] = knn[kk].second;
    }
    cout << endl;
    knn_output.push_back(row);
  }
  write_knn("knn.dat", knn_output);
  return 0;
}
