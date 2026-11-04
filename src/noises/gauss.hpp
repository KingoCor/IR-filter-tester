#ifndef GAUSS_NOISE_H
#define GAUSS_NOISE_H

#include "core/noise_model.hpp"

namespace irfilter {

	class GaussNoise : public NoiseModel {
		public:
			GaussNoise(const FrameSource& source, double sigma, std::uint64_t seed = 0)
				: NoiseModel(source, seed), sigma_(sigma) {
					CV_Assert(sigma_ > 0.0);
				}

		protected:
			cv::Mat corrupt(const cv::Mat& frame, std::uint64_t seed) const override {
				cv::RNG rng(seed);
				cv::Mat noise(frame.size(), frame.type());
				rng.fill(noise, cv::RNG::NORMAL, 0.0, sigma_);
				return frame + noise;
			}

		private:
			double sigma_;
	};

}

#endif
