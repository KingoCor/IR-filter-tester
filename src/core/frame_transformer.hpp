#ifndef FRAME_TRANSFORMER_H
#define FRAME_TRANSFORMER_H

#include "core/frame_source.hpp"

namespace irfilter {
	class FrameTransformer : public FrameSource {
		public:
			explicit FrameTransformer(const FrameSource& source) : source_(source) {}

			std::size_t size() const noexcept override { return source_.size(); }

			cv::Mat at(std::size_t index) const final {
				return transform(source_.at(index), index);
			}

		protected:
			virtual cv::Mat transform(const cv::Mat& frame, std::size_t index) const = 0;

			const FrameSource& source() const noexcept { return source_; }

		private:
			const FrameSource& source_;
	};

}

#endif
