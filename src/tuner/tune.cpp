#include "tuner_common.h"

#include <falconn/core/sketches.h>

#include <falconn/lsh_nn_table.h>

#include <Eigen/Dense>

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <cstdint>

using std::cout;
using std::endl;
using std::lock_guard;
using std::make_pair;
using std::mt19937_64;
using std::mutex;
using std::numeric_limits;
using std::ofstream;
using std::pair;
using std::runtime_error;
using std::set;
using std::shared_ptr;
using std::thread;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::time_point;

using Eigen::Map;
using Eigen::VectorXf;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborTable;
using falconn::PlainArrayPointSet;
using falconn::QueryStatistics;
using falconn::StorageHashTable;

using falconn::core::PlainArrayDataStorage;
using falconn::core::RandomProjectionSketches;
using falconn::core::RandomProjectionSketchesQuery;

using falconn::tuner::read_dataset;
using falconn::tuner::read_knn;

mutex hack;

void worker(
    shared_ptr<LSHNearestNeighborQuery<DenseVector<float>>> query_object,
    const VectorXf &center, PlainArrayPointSet<float> queries,
    int32_t query_begin, int32_t query_end, int32_t k,
    vector<vector<int32_t>> *output_knn) {
  int32_t d = queries.dimension;
  vector<int32_t> row;
  for (int32_t i = query_begin; i < query_end; ++i) {
    query_object->find_k_nearest_neighbors(
        Map<const VectorXf>(queries.data + i * d, d) - center, k, &row);
    {
      lock_guard<mutex> lock(hack);
      (*output_knn)[i] = row;
    }
  }
}

double evaluate(PlainArrayPointSet<float> points, const VectorXf &center,
                PlainArrayPointSet<float> queries, int32_t num_tables,
                int32_t num_bits, const vector<vector<uint32_t>> &knn,
                ofstream *log) {
  cout << num_bits << " bits" << endl;
  cout << num_tables << " tables" << endl;

  LSHConstructionParameters params;
  params.dimension = points.dimension;
  params.lsh_family = LSHFamily::CrossPolytope;
  params.distance_function = DistanceFunction::EuclideanSquared;
  params.l = num_tables;
  params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
  params.num_setup_threads = 0;
  params.seed = 17239;
  params.num_rotations = 1;

  compute_number_of_hash_functions<DenseVector<float>>(num_bits, &params);

  cout << "building index" << endl;
  time_point<high_resolution_clock> t1 = high_resolution_clock::now();
  auto table =
      construct_table<DenseVector<float>, int32_t, PlainArrayPointSet<float>>(
          points, params);
  time_point<high_resolution_clock> t2 = high_resolution_clock::now();
  cout << "done" << endl;
  double build_time = duration_cast<duration<double>>(t2 - t1).count();
  cout << "time: " << build_time << endl;

  uint32_t k = knn[0].size();
  vector<vector<int32_t>> output_knn(queries.num_points, vector<int32_t>(k));

  uint32_t s = 0;
  while ((int64_t(1) << s) < points.num_points) {
    ++s;
  }
  double table_size =
      double((uint64_t(1) << num_bits) + points.num_points) * s / 8.0;

  int32_t model_num_correct = -1;
  for (int32_t num_threads = 1; num_threads <= 4; ++num_threads) {
    cout << "querying from " << num_threads << " threads" << endl;
    vector<int32_t> start(num_threads + 1);
    start[0] = 0;
    for (int32_t i = 0; i < num_threads; ++i) {
      int32_t amount = queries.num_points / num_threads;
      if (i < queries.num_points % num_threads) {
        ++amount;
      }
      start[i + 1] = start[i] + amount;
    }
    vector<shared_ptr<LSHNearestNeighborQuery<DenseVector<float>>>>
        query_objects;
    for (int32_t i = 0; i < num_threads; ++i) {
      query_objects.push_back(table->construct_query_object());
    }
    vector<thread> threads;
    t1 = high_resolution_clock::now();
    for (int32_t i = 0; i < num_threads; ++i) {
      threads.push_back(thread(worker, query_objects[i], center, queries,
                               start[i], start[i + 1], k, &output_knn));
    }
    for (int32_t i = 0; i < num_threads; ++i) {
      threads[i].join();
    }
    t2 = high_resolution_clock::now();
    double query_time =
        duration_cast<duration<double>>(t2 - t1).count() / queries.num_points;
    cout << "query time: " << query_time << endl;
    int32_t num_correct = 0;
    for (int32_t i = 0; i < queries.num_points; ++i) {
      set<int32_t> s(output_knn[i].begin(), output_knn[i].end());
      for (uint32_t j = 0; j < k; ++j) {
        if (s.count(knn[i][j])) {
          ++num_correct;
        }
      }
    }
    double score = double(num_correct) / double(k * queries.num_points);
    cout << "score: " << score << endl;
    if (model_num_correct == -1) {
      model_num_correct = num_correct;
    } else if (model_num_correct != num_correct) {
      throw runtime_error("different score for multi-core");
    }
    (*log) << "num_threads " << num_threads << endl;
    (*log) << "num_bits " << num_bits << endl;
    (*log) << "num_tables " << num_tables << endl;
    (*log) << "preprocessing_time " << build_time << endl;
    (*log) << "outside_average_query_time " << query_time << endl;
    (*log) << "accuracy " << score << endl;
    (*log) << "index_size " << table_size * num_tables << endl;
    QueryStatistics query_statistics;
    for (int32_t i = 0; i < num_threads; ++i) {
      QueryStatistics q = query_objects[i]->get_query_statistics();
      q.convert_to_totals();
      query_statistics.add_totals(q);
    }
    query_statistics.compute_averages();
    (*log) << "average_total_query_time "
           << query_statistics.average_total_query_time << endl;
    (*log) << "average_lsh_time " << query_statistics.average_lsh_time << endl;
    (*log) << "average_hash_table_time "
           << query_statistics.average_hash_table_time << endl;
    (*log) << "average_distance_time " << query_statistics.average_distance_time
           << endl;
    (*log) << "average_num_candidates "
           << query_statistics.average_num_candidates << endl;
    (*log) << "average_num_unique_candidates "
           << query_statistics.average_num_unique_candidates << endl;
    (*log) << "---" << endl;
  }
  return double(model_num_correct) / double(k * queries.num_points);
}

int main() {
  vector<VectorXf> dataset, queries;
  vector<vector<uint32_t>> knn;
  read_dataset("dataset_filtered.dat", &dataset);
  read_dataset("queries.dat", &queries);
  read_knn("knn.dat", &knn);

  uint32_t n = dataset.size();
  uint32_t d = dataset[0].size();
  uint32_t q = queries.size();
  uint32_t k = knn[0].size();
  if (queries[0].size() != d) {
    throw runtime_error("wrong dimension");
  }
  if (knn.size() != q) {
    throw runtime_error("wrong number of queries");
  }
  cout << n << " " << d << " " << q << " " << k << endl;

  float *dataset_flat = new float[n * d];
  float *queries_flat = new float[q * d];

  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < d; ++j) {
      dataset_flat[i * d + j] = dataset[i][j];
    }
  }
  for (uint32_t i = 0; i < q; ++i) {
    for (uint32_t j = 0; j < d; ++j) {
      queries_flat[i * d + j] = queries[i][j];
    }
  }

  {
    vector<VectorXf> dataset_dummy, queries_dummy;
    dataset_dummy.swap(dataset);
    queries_dummy.swap(queries);
  }

  uint64_t dataset_size = n * d * 4;
  cout << "dataset size: " << dataset_size << " bytes" << endl;

  ofstream log("log.txt");
  cout << "trying linear scan (1 thread)" << endl;
  time_point<high_resolution_clock> t1 = high_resolution_clock::now();
  for (uint32_t i = 0; i < q; ++i) {
    Map<VectorXf> cur_q(queries_flat + i * d, d);
    vector<pair<float, uint32_t>> cur_knn(
        k, make_pair(numeric_limits<float>::max(), n));
    pair<float, uint32_t> best(numeric_limits<float>::max(), 0);
    for (uint32_t j = 0; j < n; ++j) {
      Map<VectorXf> cur_d(dataset_flat + j * d, d);
      float score = (cur_q - cur_d).norm();
      if (score >= best.first) {
        continue;
      }
      cur_knn[best.second] = make_pair(score, j);
      best = make_pair(-1.0, k);
      for (uint32_t kk = 0; kk < k; ++kk) {
        if (cur_knn[kk].first > best.first) {
          best = make_pair(cur_knn[kk].first, kk);
        }
      }
    }
    sort(cur_knn.begin(), cur_knn.end());
    for (uint32_t kk = 0; kk < k; ++kk) {
      if (cur_knn[kk].second != knn[i][kk]) {
        throw runtime_error("wrong answer");
      }
    }
  }
  time_point<high_resolution_clock> t2 = high_resolution_clock::now();
  cout << "done" << endl;
  double linear_scan_time =
      duration_cast<duration<double>>(t2 - t1).count() / q;
  cout << "query time: " << linear_scan_time << endl;

  cout << "centering" << endl;
  VectorXf center(VectorXf::Zero(d));
  for (size_t i = 0; i < n; ++i) {
    center += Map<VectorXf>(dataset_flat + i * d, d);
  }
  center /= n;
  cout << center.norm() << endl;
  for (size_t i = 0; i < n; ++i) {
    Map<VectorXf>(dataset_flat + i * d, d) -= center;
  }
  cout << "done" << endl;

  cout << "optimized linear scan (1 thread)" << endl;
  PlainArrayDataStorage<DenseVector<float>> pads(dataset_flat, n, d);
  mt19937_64 gen(612534);
  for (int32_t num_chunks = 1; num_chunks <= 50; ++num_chunks) {
    cout << 64 * num_chunks << " bits" << endl;
    RandomProjectionSketches<DenseVector<float>,
                             PlainArrayDataStorage<DenseVector<float>>>
        sketches(pads, num_chunks, gen);
    RandomProjectionSketchesQuery<DenseVector<float>, int32_t,
                                  PlainArrayDataStorage<DenseVector<float>>>
        sketches_query(sketches);
    vector<int32_t> distances;
    for (uint32_t i = 0; i < q; ++i) {
      sketches_query.load_query(Map<VectorXf>(queries_flat + i * d, d) -
                                center);
      for (uint32_t j = 0; j < k; ++j) {
        distances.push_back(sketches_query.get_distance_estimate(knn[i][j]));
      }
    }
    sort(distances.begin(), distances.end());
    int32_t threshold = distances[(size_t)ceil(0.9 * k * q) - 1];
    cout << "threshold: " << threshold << endl;
    sketches_query.set_distance_threshold(threshold);
    int64_t num_candidates = 0;
    t1 = high_resolution_clock::now();
    vector<vector<uint32_t>> output_knn(q, vector<uint32_t>(k));
    for (uint32_t i = 0; i < q; ++i) {
      VectorXf cur_q = Map<VectorXf>(queries_flat + i * d, d) - center;
      sketches_query.load_query(cur_q);
      vector<pair<float, uint32_t>> cur_knn(
          k, make_pair(numeric_limits<float>::max(), n));
      pair<float, uint32_t> best(numeric_limits<float>::max(), 0);
      for (uint32_t j = 0; j < n; ++j) {
        if (!sketches_query.is_close(j)) {
          continue;
        }
        ++num_candidates;
        Map<VectorXf> cur_d(dataset_flat + j * d, d);
        float score = (cur_q - cur_d).norm();
        if (score >= best.first) {
          continue;
        }
        cur_knn[best.second] = make_pair(score, j);
        best = make_pair(-1.0, k);
        for (uint32_t kk = 0; kk < k; ++kk) {
          if (cur_knn[kk].first > best.first) {
            best = make_pair(cur_knn[kk].first, kk);
          }
        }
      }
      for (uint32_t j = 0; j < k; ++j) {
        output_knn[i][j] = cur_knn[j].second;
      }
    }
    t2 = high_resolution_clock::now();
    int32_t num_correct = 0;
    for (uint32_t i = 0; i < q; ++i) {
      set<int32_t> s(output_knn[i].begin(), output_knn[i].end());
      for (uint32_t j = 0; j < k; ++j) {
        if (s.count(knn[i][j])) {
          ++num_correct;
        }
      }
    }
    double score = double(num_correct) / double(q * k);
    double query_time =
        duration_cast<duration<double>>(t2 - t1).count() / double(q);
    cout << "threshold: " << threshold << endl;
    cout << "candidates: " << double(num_candidates) / double(q) << endl;
    cout << "time: " << query_time << endl;
    cout << "score: " << score << endl;
  }
  cout << "done" << endl;

  PlainArrayPointSet<float> points;
  points.data = dataset_flat;
  points.num_points = n;
  points.dimension = d;

  PlainArrayPointSet<float> queries_wrapped;
  queries_wrapped.data = queries_flat;
  queries_wrapped.num_points = q;
  queries_wrapped.dimension = d;

  uint32_t s = 0;
  while ((uint64_t(1) << s) < n) {
    ++s;
  }

  cout << s << endl;

  bool phase_1 = true;
  int32_t prev_num_tables = numeric_limits<int32_t>::max();

  for (int32_t num_bits = s + 2; num_bits >= 1; --num_bits) {
    bool checked_upper_bound = false;
    if (phase_1) {
      double table_size = double((uint64_t(1) << num_bits) + n) * s / 8.0;

      int32_t max_num_tables = floor(4e9 / table_size);
      int32_t num_tables = max_num_tables;

      double score = evaluate(points, center, queries_wrapped, num_tables,
                              num_bits, knn, &log);

      if (score >= 0.9) {
        phase_1 = false;
        prev_num_tables = num_tables;
        checked_upper_bound = true;
      }
    }
    if (!phase_1) {
      if (!checked_upper_bound) {
        double score = evaluate(points, center, queries_wrapped,
                                prev_num_tables, num_bits, knn, &log);
        if (score < 0.9) {
          cout << "WARNING: insufficient upper bound" << endl;
          continue;
        }
      }
      int32_t left = 0;
      int32_t right = prev_num_tables;
      cout << "binary search between " << left << " and " << right << endl;
      for (;;) {
        if (right <= left + 1) {
          break;
        }
        if (right <= 1.1 * left) {
          break;
        }
        int32_t middle = (left + right) / 2;
        double score = evaluate(points, center, queries_wrapped, middle,
                                num_bits, knn, &log);
        if (score >= 0.9) {
          right = middle;
        } else {
          left = middle;
        }
      }
      prev_num_tables = right;
    }
  }

  delete[] dataset_flat;
  delete[] queries_flat;
  return 0;
}
