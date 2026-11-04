#pragma once

#include "core/metric.hpp"

namespace irfilter {

class MAE : public ReferenceMetric {
public:
    std::string name() const override { return "MAE"; }
    double compute(const cv::Mat& ref, const cv::Mat& proc) const override {
        return cv::norm(ref, proc, cv::NORM_L1) / static_cast<double>(ref.total() * ref.channels());
    }
};

class MSE : public ReferenceMetric {
public:
    std::string name() const override { return "MSE"; }
    double compute(const cv::Mat& ref, const cv::Mat& proc) const override {
        const cv::Mat d = ref - proc;
        return d.dot(d) / static_cast<double>(ref.total() * ref.channels());
    }
};

} // namespace irfilter
