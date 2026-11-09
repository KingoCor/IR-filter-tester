#pragma once

#include "core/metric.hpp"

#include <cmath>
#include <limits>
#include <opencv2/core.hpp>

namespace irfilter {

class PSNR : public ReferenceMetric {
public:
    explicit PSNR(double peak = 4095.0) : peak_(peak) {}

    std::string name() const override { return "PSNR"; }

    double compute(const cv::Mat& ref, const cv::Mat& proc) const override {
        // Приводим к float, чтобы не было переполнения/насыщения для целочисленных типов
        cv::Mat a, b;
        ref.convertTo(a, CV_32F);
        proc.convertTo(b, CV_32F);

        const cv::Mat d = a - b;
        const double mse = d.dot(d) / static_cast<double>(ref.total() * ref.channels());

        if (mse <= 0.0) return std::numeric_limits<double>::infinity();
        return 10.0 * std::log10(peak_ * peak_ / mse);
    }

private:
    double peak_;
};

} // namespace irfilter