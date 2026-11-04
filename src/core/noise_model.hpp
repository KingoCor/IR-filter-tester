#ifndef NOISE_MODEL_H
#define NOISE_MODEL_H

#include "core/frame_transformer.hpp"
#include <cstdint>

namespace irfilter {

	class NoiseModel : public FrameTransformer {
		public:
			explicit NoiseModel(const FrameSource& source, std::uint64_t seed = 0)
				: FrameTransformer(source), baseSeed_(seed) {}

			std::uint64_t seed() const noexcept { return baseSeed_; }

		protected:
			cv::Mat transform(const cv::Mat& frame, std::size_t index) const final;
			virtual cv::Mat corrupt(const cv::Mat& frame, std::uint64_t seed) const = 0;

		private:
			std::uint64_t baseSeed_;
	};

}

#endif
