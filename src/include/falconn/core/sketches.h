#ifndef _SKETCHES_H_
#define _SKETCHES_H_

#include "../falconn_global.h"
#include "polytope_hash.h"

#include <memory>
#include <random>
#include <vector>

#include <cstdlib>

namespace falconn {
namespace core {

class SketchesError : public FalconnError {
 public:
  SketchesError(const char *msg) : FalconnError(msg) {}
};

namespace sketches_helpers {

template <typename PointType, typename DataStorageType = std::vector<PointType>>
class RandomProjectionSketchesWorker {
 public:
  typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;

  RandomProjectionSketchesWorker(int32_t dimension, int32_t num_rotations,
                                 int32_t num_chunks,
                                 const std::vector<ScalarType> &random_signs)
      : dimension_(dimension),
        num_rotations_(num_rotations),
        num_chunks_(num_chunks),
        random_signs_(random_signs) {
    padded_dimension_ = dimension;
    while (padded_dimension_ & (padded_dimension_ - 1)) {
      ++padded_dimension_;
    }

    ScalarType *buffer;
    if (posix_memalign((void **)&buffer, 32,
                       padded_dimension_ * sizeof(ScalarType))) {
      throw SketchesError("can't allocate memory for the FHT buffer");
    }

    buffer_ = std::shared_ptr<ScalarType>(buffer, free);

    if (padded_dimension_ != dimension_) {
      padding_ = PointType::Zero(padded_dimension_ - dimension_);
    }
  }

  void compute_sketch(const PointType &point, uint64_t *result) {
    for (int32_t i = 0; i < num_rotations_; ++i) {
      Eigen::Map<PointType>(buffer_.get(), dimension_) =
          point.cwiseProduct(Eigen::Map<const PointType>(
              &random_signs_[i * dimension_], dimension_));
      if (dimension_ != padded_dimension_) {
        Eigen::Map<PointType>(buffer_.get() + dimension_,
                              padded_dimension_ - dimension_) = padding_;
      }

      cp_hash_helpers::FHTFunction<ScalarType>::apply(buffer_.get(),
                                                      padded_dimension_);

      for (int32_t j = 0; j < padded_dimension_; ++j) {
        int32_t pos = i * padded_dimension_ + j;

        int32_t chunk_id = pos / 64;
        if (chunk_id >= num_chunks_) {
          break;
        }
        uint64_t bit = 0;
        if (buffer_.get()[j] > 0.0) {
          bit = 1;
        }

        int32_t chunk_off = pos % 64;

        result[chunk_id] = (result[chunk_id] & ~(uint64_t(1) << chunk_off)) |
                           (bit << chunk_off);
      }
    }
  }

 private:
  int32_t dimension_;
  int32_t padded_dimension_;
  int32_t num_rotations_;
  int32_t num_chunks_;
  const std::vector<ScalarType> &random_signs_;
  std::shared_ptr<ScalarType> buffer_;
  PointType padding_;
};
}

template <typename PointType, typename IndexType = int32_t,
          typename DataStorageType = std::vector<PointType>>
class RandomProjectionSketchesQuery;

template <typename PointType, typename DataStorageType = std::vector<PointType>>
class RandomProjectionSketches {
 public:
  typedef typename PointTypeTraits<PointType>::ScalarType ScalarType;

  template <typename RNG>
  RandomProjectionSketches(const DataStorageType &points, int32_t num_chunks,
                           RNG &rng)
      : num_chunks_(num_chunks), sketches_(points.size() * num_chunks) {
    if (num_chunks < 1) {
      throw SketchesError("there must be at least one chunk");
    }

    typename DataStorageType::FullSequenceIterator iter =
        points.get_full_sequence();
    dimension_ = iter.get_point().size();

    int32_t num_bits = 64 * num_chunks_;

    int32_t padded_dimension = dimension_;
    while (padded_dimension & (padded_dimension - 1)) {
      ++padded_dimension;
    }

    num_rotations_ = num_bits / padded_dimension;
    if (num_bits % padded_dimension) {
      ++num_rotations_;
    }

    random_signs_.resize(num_rotations_ * dimension_);

    std::uniform_int_distribution<> random_bit(0, 1);

    for (int32_t i = 0; i < num_rotations_ * dimension_; ++i) {
      random_signs_[i] = 1.0 - 2.0 * random_bit(rng);
    }

    sketches_helpers::RandomProjectionSketchesWorker<PointType, DataStorageType>
        worker(dimension_, num_rotations_, num_chunks_, random_signs_);

    size_t counter = 0;
    while (iter.is_valid()) {
      worker.compute_sketch(iter.get_point(),
                            &sketches_[counter * num_chunks_]);
      ++counter;
      ++iter;
    }
  }

 private:
  int32_t num_chunks_;
  std::vector<uint64_t> sketches_;
  std::vector<ScalarType> random_signs_;

  int32_t dimension_;
  int32_t num_rotations_;

  template <typename, typename, typename>
  friend class RandomProjectionSketchesQuery;
};

template <typename PointType, typename IndexType, typename DataStorageType>
class RandomProjectionSketchesQuery {
 public:
  RandomProjectionSketchesQuery(
      const RandomProjectionSketches<PointType, DataStorageType> &sketch,
      int32_t distance_threshold = -1)
      : sketch_(sketch),
        num_chunks_(sketch.num_chunks_),
        distance_threshold_(distance_threshold),
        worker_(sketch.dimension_, sketch.num_rotations_, sketch.num_chunks_,
                sketch.random_signs_),
        query_sketch_(sketch.num_chunks_) {
    if (distance_threshold_ < -1) {
      throw SketchesError("distance threshold must be -1 or non-negative");
    }
    if (distance_threshold_ == -1) {
      distance_threshold_ = 64 * num_chunks_;
    }
  }

  void load_query(const PointType &query) {
    worker_.compute_sketch(query, &query_sketch_[0]);
    loaded_ = true;
  }

  inline int32_t get_distance_estimate(IndexType dataset_point_id) {
    if (!loaded_) {
      throw SketchesError("query is not loaded");
    }
    int32_t hamming_distance = 0;
    size_t ind = dataset_point_id * num_chunks_;
    for (int32_t i = 0; i < num_chunks_; ++i) {
      hamming_distance +=
          __builtin_popcountll(sketch_.sketches_[ind] ^ query_sketch_[i]);
      ++ind;
    }
    return hamming_distance;
  }

  inline bool is_close(IndexType dataset_point_id) {
    return get_distance_estimate(dataset_point_id) <= distance_threshold_;
  }

  inline void filter_close(const std::vector<IndexType> &candidates,
                           std::vector<IndexType> *filtered_candidates) {
    filtered_candidates->clear();
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (is_close(candidates[i])) {
        filtered_candidates->push_back(candidates[i]);
      }
    }
  }

  void set_distance_threshold(int32_t threshold) {
    distance_threshold_ = threshold;
  }

 private:
  const RandomProjectionSketches<PointType, DataStorageType> &sketch_;
  int32_t num_chunks_;
  int32_t distance_threshold_;
  sketches_helpers::RandomProjectionSketchesWorker<PointType, DataStorageType>
      worker_;

  std::vector<uint64_t> query_sketch_;
  bool loaded_ = false;
};
}
}

#endif
