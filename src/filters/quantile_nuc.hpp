#pragma once

#include "core/frame_transformer.hpp"

#include <vector>

namespace irfilter {

	class QuantileNUC : public FrameTransformer {
		public:
			struct Params {
				int num_quantiles = 16;   // N: число интервалов, маркеров N+1
				int max_frames    = 0;    // сколько кадров использовать для оценки (0 = все)
			};

			explicit QuantileNUC(const FrameSource& source);
			QuantileNUC(const FrameSource& source, Params params);

		protected:
			cv::Mat transform(const cv::Mat& frame, std::size_t index) const override;

		private:
			void initialize();

			Params params_;
			int W_ = 0, H_ = 0;

			// q_[pix*(N+1) + m] — m-й маркер (квантиль m/N) пикселя pix, по возрастанию
			std::vector<float> q_;
	};

}