#pragma once

#include "core/frame_transformer.hpp"

#include <string>
#include <vector>

namespace irfilter {

	class PaperNUC : public FrameTransformer {
		public:
			struct Params {
				int    frame_buff_size = 360;
				int    block_size_L = 20;
				double segment_threshold = 10.0 / 4095.0;

				std::string dump_dir;
			};

			explicit PaperNUC(const FrameSource& source);
			PaperNUC(const FrameSource& source, Params params);

			cv::Mat ExtractXOY(int row);
			void SortSurfaceByRows(cv::Mat &surface);
			void initialize();

		protected:
			cv::Mat transform(const cv::Mat& frame, std::size_t index) const override;

		private:
			Params params_;

			struct Segment {
				double value_min, value_max;
				double k, b;
			};

			std::vector<std::vector<std::vector<Segment>>> coeffs_;
	};

} // namespace irfilter
