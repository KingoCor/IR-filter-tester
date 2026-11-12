#pragma once

#include "core/frame_transformer.hpp"

#include <limits>
#include <string>
#include <vector>

namespace irfilter {

    class QuantileNUC : public FrameTransformer {
        public:
            struct Params {
                int    num_quantiles = 7;    // N — число квантилей
                int    warmup_frames = 800;     // 0 = полностью онлайн с 0-го кадра
                double eta0 = 0.005; // η в нормированном домене [0,1]
                double eta_T = 800;
				int quantile_step = 1;

				int    reference_rows = 3;

                // Опционально: зафиксировать глобальный диапазон
                double global_min    = std::numeric_limits<double>::quiet_NaN();
                double global_max    = std::numeric_limits<double>::quiet_NaN();

                std::string dump_dir;
            };

            explicit QuantileNUC(const FrameSource& source);
            QuantileNUC(const FrameSource& source, Params params);

            int  warmupFrames()    const { return consumed_warmup_; }
            bool isWarmedUp()      const { return static_cast<int>(size()) <= consumed_warmup_; }
            void reset();

        protected:
            cv::Mat transform(const cv::Mat& frame, std::size_t index) const override;

        private:
            void initQuantilesLinear() const;
            void updateQuantiles    (const cv::Mat& frame32, int index) const;
            void computeReference   () const;
            cv::Mat applyTransform  (const cv::Mat& frame32) const;

            Params params_;

            int    H_ = 0, W_ = 0, N_ = 0;
            double gmin_ = 0.0, gmax_ = 1.0;
			int frames_seen_ = 0;

            // mutable — обновляется в const transform()
            mutable int consumed_warmup_ = 0;   // сколько первых кадров уже «съедено»
            mutable int last_index_      = -1;

            mutable std::vector<std::vector<std::vector<float>>> quantiles_; // [H][W][N] в [0,1]
            mutable std::vector<std::vector<float>>              reference_; // [H][N]    в [0,1]
    };

} // namespace irfilter