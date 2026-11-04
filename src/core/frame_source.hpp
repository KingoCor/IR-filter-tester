#pragma once

#include <opencv2/core.hpp>
#include <cstddef>

namespace irfilter {
	class FrameSource {
		public:
			virtual ~FrameSource() = default;

			virtual std::size_t size() const noexcept = 0;
			virtual cv::Mat     at(std::size_t index) const = 0;
	};
}
