#include "tuner_common.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cstdint>

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::runtime_error;
using std::string;
using std::stringstream;
using std::vector;

using Eigen::VectorXf;

using falconn::tuner::write_dataset;

int main() {
    uint32_t d = 100;
    ifstream input("glove.twitter.27B.100d.txt");
    uint32_t n = 0;
    vector<VectorXf> dataset;
    for (;;) {
        string s;
        if (!getline(input, s)) {
            break;
        }
        ++n;
        stringstream ss(s);
        string dummy;
        ss >> dummy;
        float t;
        uint32_t cnt = 0;
        VectorXf row(d);
        while (ss >> t) {
            if (cnt >= d) {
                throw runtime_error("wrong dimension");
            }
            row[cnt] = t;
            ++cnt;
        }
        if (cnt != d) {
            throw runtime_error("wrong dimension");
        }
        row.normalize();
        dataset.push_back(row);
    }
    cout << n << " " << d << endl;
    write_dataset("dataset.dat", dataset);
    return 0;
}
