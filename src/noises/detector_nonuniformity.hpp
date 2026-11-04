#pragma once

#include "core/noise_model.hpp"
#include <vector>

namespace irfilter {

	class DetectorNonuniformityNoise : public NoiseModel {
		public:
			struct Params {
				float A_min =  0.80f, A_max = 1.00f;
				float B_min =  5.00f, B_max = 10.0f;
				float C_min =  0.40f, C_max = 0.60f;
				float D_min = -0.05f, D_max = 0.05f;
			};

			explicit DetectorNonuniformityNoise(const FrameSource& source, std::uint64_t seed = 0);
			DetectorNonuniformityNoise(const FrameSource& source, Params params, std::uint64_t seed = 0);

		protected:
			cv::Mat corrupt(const cv::Mat& frame, std::uint64_t seed) const override;

		private:
			void initialize(int H, int W) const;

			Params params_;

			mutable bool initialized_ = false;
			mutable cv::Mat A_, B_, C_, D_;
			mutable std::vector<float> A_n_, B_n_, C_n_, D_n_;
	};

} // namespace irfilter
