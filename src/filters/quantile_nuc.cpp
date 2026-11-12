#include "filters/quantile_nuc.hpp"
#include "core/frame_source.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace irfilter {

    QuantileNUC::QuantileNUC(const FrameSource& source)
        : QuantileNUC(source, Params{}) {}

    QuantileNUC::QuantileNUC(const FrameSource& source, Params params)
        : FrameTransformer(source), params_(params)
    {
        if (params_.num_quantiles < 2)
            throw std::runtime_error("QuantileNUC: num_quantiles must be >= 2");
        if (params_.warmup_frames < 0)
            throw std::runtime_error("QuantileNUC: warmup_frames must be >= 0");
        if (source.size() < 1)
            throw std::runtime_error("QuantileNUC: need at least 1 frame");

        // ---- 1. Геометрия из первого кадра ----
        const cv::Mat& first = source.at(0);
        if (first.channels() != 1)
            throw std::runtime_error("QuantileNUC: expected single-channel frames");
        H_ = first.rows;
        W_ = first.cols;
        N_ = params_.num_quantiles;

        // ---- 2. Глобальный диапазон ----
        if (std::isfinite(params_.global_min) &&
            std::isfinite(params_.global_max) &&
            params_.global_max > params_.global_min)
        {
            gmin_ = params_.global_min;
            gmax_ = params_.global_max;
        } else {
            // Сканируем первые max(1, warmup) кадров для диапазона
            const int scan_n = std::min<int>(
                std::max(1, params_.warmup_frames),
                static_cast<int>(source.size()));
            double mn =  std::numeric_limits<double>::max();
            double mx = -std::numeric_limits<double>::max();
            for (int f = 0; f < scan_n; ++f) {
                cv::Mat f32;
                source.at(f).convertTo(f32, CV_32F);
                double fmn, fmx;
                cv::minMaxLoc(f32, &fmn, &fmx);
                mn = std::min(mn, fmn);
                mx = std::max(mx, fmx);
            }
            gmin_ = mn;
            gmax_ = std::max(mx, mn + 1e-6);
        }

        // ---- 3. Квантили = прямая q_i = p_i ----
        initQuantilesLinear();

        // ---- 4. Опциональный препрогон по первым n кадрам ----
        const int warmup = params_.warmup_frames;
        for (int f = 0; f < warmup; ++f) {
			std::cout << "Intialization: " << f << "/" << warmup << "\n";
            cv::Mat f32;
            source.at(f%source.size()).convertTo(f32, CV_32F);
            updateQuantiles(f32, f);
            ++consumed_warmup_;
        }
        if (consumed_warmup_ > 0)
            computeReference();
    }

    void QuantileNUC::reset() {
        initQuantilesLinear();
        if (consumed_warmup_ > 0)
            computeReference();
        else
            reference_.clear();
        last_index_ = -1;
    }

    // -------------------------------------------------------------------------
    // q_i = p_i = (i + 0.5) / N — равномерная прямая в (0, 1).
    // -------------------------------------------------------------------------
    void QuantileNUC::initQuantilesLinear() const {
        const float Nf = static_cast<float>(N_);
        quantiles_.assign(H_, std::vector<std::vector<float>>(
                                   W_, std::vector<float>(N_, 0.0f)));
        for (int r = 0; r < H_; ++r)
            for (int w = 0; w < W_; ++w)
                for (int i = 0; i < N_; ++i)
                    quantiles_[r][w][i] = (static_cast<float>(i) + 0.5f) / Nf;
    }

    // -------------------------------------------------------------------------
    // Потоковое обновление в нормированном домене:
    //   q_i <- q_i + η·(p_i − I(x_n ≤ q_i))
    // плюс sort — проекция на монотонный конус.
    // -------------------------------------------------------------------------
    void QuantileNUC::updateQuantiles(const cv::Mat& frame32, int index) const {
        const float inv_range = 1.0f / static_cast<float>(gmax_ - gmin_);
        const float gminf     = static_cast<float>(gmin_);
        const float Nf        = static_cast<float>(N_);

		const double t  = static_cast<double>(index);
		const double eta_f = params_.eta0/(1.0+t/params_.eta_T);
		const float  eta   = static_cast<float>(eta_f * (gmax_ - gmin_));

        for (int r = 0; r < H_; ++r) {
            const float* src = frame32.ptr<float>(r);
            for (int w = 0; w < W_; ++w) {
                const float x = (src[w] - gminf) * inv_range;
                auto& q = quantiles_[r][w];

                for (int i = 0; i < N_; ++i) {
                    const float p_i = (static_cast<float>(i) + 0.5f) / Nf;
                    const float I   = (x <= q[i]) ? 1.0f : 0.0f;
                    q[i] += eta * (p_i - I);
                }
                //std::sort(q.begin(), q.end()); // монотонность
            }
        }
    }

	void QuantileNUC::computeReference() const {
		const int half = std::max(0, params_.reference_rows / 2);

		reference_.assign(H_, std::vector<float>(N_, 0.0f));

		// 1. Строковые суммы по квантилям
		std::vector<std::vector<double>> rowsum(N_, std::vector<double>(H_, 0.0));
		for (int r = 0; r < H_; ++r) {
			for (int w = 0; w < W_; ++w) {
				const auto& q = quantiles_[r][w];
				for (int i = 0; i < N_; ++i)
					rowsum[i][r] += q[i];
			}
		}

		// 2. Префиксные суммы по строкам
		std::vector<std::vector<double>> prefix(N_, std::vector<double>(H_ + 1, 0.0));
		for (int i = 0; i < N_; ++i)
			for (int r = 0; r < H_; ++r)
				prefix[i][r + 1] = prefix[i][r] + rowsum[i][r];

		// 3. Скользящее окно
		for (int r = 0; r < H_; ++r) {
			const int r0 = std::max(0, r - half);
			const int r1 = std::min(H_, r + half + 1);
			const double inv = 1.0 / static_cast<double>((r1 - r0) * W_);
			for (int i = 0; i < N_; ++i)
				reference_[r][i] = static_cast<float>((prefix[i][r1] - prefix[i][r0]) * inv);
		}
	}

    // -------------------------------------------------------------------------
    // Кусочно-линейное отображение q_i → ref_i по парам
    // (q_i, ref_i), (q_{i+1}, ref_{i+1}).
    // -------------------------------------------------------------------------
    cv::Mat QuantileNUC::applyTransform(const cv::Mat& frame32) const {
        cv::Mat out(frame32.size(), CV_32F);
        const float range     = static_cast<float>(gmax_ - gmin_);
        const float inv_range = 1.0f / range;
        const float gminf     = static_cast<float>(gmin_);
        const float eps       = 1e-6f;

        for (int r = 0; r < frame32.rows; ++r) {
            const float* src = frame32.ptr<float>(r);
            float*       dst = out.ptr<float>(r);
            const auto&  ref = reference_[r];

            for (int w = 0; w < frame32.cols; ++w) {
                const float x_n = (src[w] - gminf) * inv_range;
                const auto& q   = quantiles_[r][w];

                auto it = std::upper_bound(q.begin(), q.end(), x_n);
                int  i  = static_cast<int>(it - q.begin()) - 1;
				i = std::clamp(i, 0, N_ - 1 - params_.quantile_step);
				const int j = i + params_.quantile_step;

                const float q0 = q[i],   q1 = q[j];
                const float r0 = ref[i], r1 = ref[j];
                const float dq = q1 - q0;

                float y_n;
                if (std::abs(dq) < eps) {
                    y_n = x_n + (r0 - q0);                 // вырожденный сегмент — сдвиг
                } else {
                    const float k = (r1 - r0) / dq;
                    const float b = r0 - k * q0;
                    y_n = k * x_n + b;
                }
                dst[w] = y_n * range + gminf;
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Главный вход. Один вызов = один кадр потока.
    // -------------------------------------------------------------------------
    cv::Mat QuantileNUC::transform(const cv::Mat& frame, std::size_t index) const {
        cv::Mat frame32;
        frame.convertTo(frame32, CV_32F);

        updateQuantiles(frame32, static_cast<int>(index)+params_.warmup_frames);
        computeReference();
        last_index_ = static_cast<int>(index);

        return applyTransform(frame32);
    }

} // namespace irfilter