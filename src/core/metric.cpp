#include "core/metric.hpp"
#include "core/frame_source.hpp"

#include <algorithm>
#include <stdexcept>

namespace irfilter {

	double aggregate(const std::vector<double>& v, Aggregation how) {
		if (v.empty()) throw std::runtime_error("aggregate: empty input");
		switch (how) {
			case Aggregation::Mean: {
				double s = 0.0;
				for (double x : v) s += x;
				return s / static_cast<double>(v.size());
			}
			case Aggregation::Median: {
				auto tmp = v;
				const auto mid = tmp.begin() + static_cast<std::ptrdiff_t>(tmp.size() / 2);
				std::nth_element(tmp.begin(), mid, tmp.end());
				const double hi = *mid;
				if (tmp.size() % 2 == 1) return hi;
				const double lo = *std::max_element(tmp.begin(), mid);
				return 0.5 * (lo + hi);
			}
			case Aggregation::Min: return *std::min_element(v.begin(), v.end());
			case Aggregation::Max: return *std::max_element(v.begin(), v.end());
		}
		throw std::logic_error("aggregate: unknown Aggregation");
	}

	double evaluate(
		const ReferenceMetric& metric,
		const FrameSource& reference,
		const FrameSource& processed,
		Aggregation agg
	) {
		const std::size_t n = reference.size();
		if (n != processed.size()) {
			throw std::runtime_error(metric.name() + ": stream sizes differ (" +
					std::to_string(n) + " vs " +
					std::to_string(processed.size()) + ")");
		}

		std::vector<double> values;
		values.reserve(n);
		for (std::size_t i = 0; i < n; ++i) {
			const cv::Mat ref = reference.at(i);
			const cv::Mat prc = processed.at(i);
			if (ref.size() != prc.size() || ref.type() != prc.type()) {
				throw std::runtime_error(metric.name() + ": frame " + std::to_string(i) +
						" mismatch (size or type)");
			}
			values.push_back(metric.compute(ref, prc));
		}
		return aggregate(values, agg);
	}

	double evaluate(
		const NoReferenceMetric& metric,
		const FrameSource& processed,
		Aggregation agg
	) {
		const std::size_t n = processed.size();
		std::vector<double> values;
		values.reserve(n);
		for (std::size_t i = 0; i < n; ++i) {
			values.push_back(metric.compute(processed.at(i)));
		}
		return aggregate(values, agg);
	}

}
