#pragma once

#include "core/frame_transformer.hpp"

#include <vector>

namespace irfilter {

	class ColumnHistNUC : public FrameTransformer {
		public:
			struct Params {
				int    num_bins     = 4096;   // число бинов гистограммы столбца
				double max_value    = 1.0;    // максимум входной шкалы (1.0 для нормированных кадров)
				int    num_quant    = 1024;   // число уровней u_k в обратной CDF
				int    radius       = 4;      // соседей с каждой стороны
				double sigma        = 2.0;    // сигма гауссовых весов (в столбцах)
				bool   include_self = false;  // учитывать ли сам столбец в целевой H^-1
				int    max_frames   = 0;      // сколько кадров использовать (0 = все)
			};

			explicit ColumnHistNUC(const FrameSource& source);
			ColumnHistNUC(const FrameSource& source, Params params);

		protected:
			cv::Mat transform(const cv::Mat& frame, std::size_t index) const override;

		private:
			void initialize();

			Params params_;
			int W_ = 0, H_ = 0;
			float bin_scale_ = 0.f;   // num_bins / max_value

			// lut_[x * num_bins + bin] — скорректированное значение для столбца x
			std::vector<float> lut_;
	};

}