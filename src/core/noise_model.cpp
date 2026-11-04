#include "core/noise_model.hpp"

namespace irfilter {

	cv::Mat NoiseModel::transform(const cv::Mat& frame, std::size_t index) const {
		cv::Mat f;
		if (frame.depth() != CV_32F) {
			frame.convertTo(f, CV_32F);
		} else {
			f = frame;
		}

		const std::uint64_t s = baseSeed_ + static_cast<std::uint64_t>(index);
		cv::Mat out = corrupt(f, s);

		CV_Assert(out.depth() == CV_32F);
		CV_Assert(out.size() == f.size());
		CV_Assert(out.channels() == f.channels());

		return out;
	}

}
