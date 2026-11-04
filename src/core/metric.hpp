#ifndef METRIC_H
#define METRIC_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstddef>

namespace irfilter {

	class FrameSource;

	class Metric {
		public:
			virtual ~Metric() = default;
			virtual std::string name() const = 0;
	};

	class ReferenceMetric : public Metric {
		public:
			virtual double compute(const cv::Mat& reference, const cv::Mat& processed) const = 0;
	};

	class NoReferenceMetric : public Metric {
		public:
			virtual double compute(const cv::Mat& processed) const = 0;
	};

	enum class Aggregation { Mean, Median, Min, Max };

	double aggregate(const std::vector<double>& values, Aggregation how);

	double evaluate(
		const ReferenceMetric& metric,
		const FrameSource& reference,
		const FrameSource& processed,
		Aggregation agg = Aggregation::Mean
	);

	double evaluate(
		const NoReferenceMetric& metric,
		const FrameSource& processed,
		Aggregation agg = Aggregation::Mean
	);

}

#endif
