#include "filters/quantile_nuc.hpp"
#include "core/frame_source.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace irfilter {

	QuantileNUC::QuantileNUC(const FrameSource& source) : QuantileNUC(source, Params{}) {}

	QuantileNUC::QuantileNUC(const FrameSource& source, Params params)
		: FrameTransformer(source), params_(params) {
		if (params_.num_quantiles < 2) throw std::runtime_error("QuantileNUC: num_quantiles must be >= 2");
		if (source.size() < static_cast<std::size_t>(params_.num_quantiles + 1))
			throw std::runtime_error("QuantileNUC: need at least N+1 frames");
		initialize();
	}

	void QuantileNUC::initialize() {
		const cv::Mat first = source().at(0);
		if (first.channels() != 1) throw std::runtime_error("QuantileNUC::initialize: expected single-channel frames");

		W_ = first.cols; H_ = first.rows;
		const int HW = W_ * H_;
		const int N  = params_.num_quantiles;
		const int M  = N + 1;

		int F = static_cast<int>(size());
		if (params_.max_frames > 0) F = std::min(F, params_.max_frames);
		if (F < M) throw std::runtime_error("QuantileNUC::initialize: not enough frames");

		q_.assign(static_cast<std::size_t>(HW) * M, 0.f);
		std::vector<int> n(static_cast<std::size_t>(HW) * M);

		// 1) Инициализация по первым M кадрам: маркеры = отсортированные M значений, позиции 1..M
		{
			std::vector<cv::Mat> init(M);
			for (int f = 0; f < M; ++f) {
				source().at(f).convertTo(init[f], CV_32F);
				if (!init[f].isContinuous()) init[f] = init[f].clone();
			}
			cv::parallel_for_(cv::Range(0, HW), [&](const cv::Range& rg) {
				for (int i = rg.start; i < rg.end; ++i) {
					float* qi = &q_[static_cast<std::size_t>(i) * M];
					int*   ni = &n[static_cast<std::size_t>(i) * M];
					for (int f = 0; f < M; ++f) qi[f] = init[f].ptr<float>()[i];
					std::sort(qi, qi + M);
					for (int m = 0; m < M; ++m) ni[m] = m + 1;
				}
			});
		}

		// 2) Потоковое обновление P² по остальным кадрам
		for (int f = M; f < F; ++f) {
			cv::Mat fr;
			source().at(f).convertTo(fr, CV_32F);
			if (!fr.isContinuous()) fr = fr.clone();
			const float* px = fr.ptr<float>();
			const int t = f + 1;   // число уже просмотренных наблюдений (с текущим)

			cv::parallel_for_(cv::Range(0, HW), [&](const cv::Range& rg) {
				for (int i = rg.start; i < rg.end; ++i) {
					float* q = &q_[static_cast<std::size_t>(i) * M];
					int*   p = &n[static_cast<std::size_t>(i) * M];
					const float x = px[i];

					// ячейка k: q[k] <= x < q[k+1]
					int k;
					if (x < q[0])      { q[0] = x; k = 0; }
					else if (x >= q[N]) { q[N] = x; k = N - 1; }
					else {
						k = static_cast<int>(std::upper_bound(q, q + M, x) - q) - 1;
						if (k > N - 1) k = N - 1;
					}
					for (int m = k + 1; m < M; ++m) ++p[m];

					// подгонка внутренних маркеров к желаемым позициям
					for (int m = 1; m < N; ++m) {
						const double desired = 1.0 + (t - 1) * static_cast<double>(m) / N;
						const double d = desired - p[m];

						if ((d >= 1.0 && p[m + 1] - p[m] > 1) ||
							(d <= -1.0 && p[m - 1] - p[m] < -1)) {
							const int s = d >= 0 ? 1 : -1;

							const double dn1 = p[m + 1] - p[m];
							const double dn0 = p[m] - p[m - 1];
							const double qp = q[m] + s / static_cast<double>(p[m + 1] - p[m - 1]) *
								((dn0 + s) * (q[m + 1] - q[m]) / dn1 +
								 (dn1 - s) * (q[m] - q[m - 1]) / dn0);

							if (qp > q[m - 1] && qp < q[m + 1]) {
								q[m] = static_cast<float>(qp);
							} else {  // линейная
								q[m] += static_cast<float>(s * (q[m + s] - q[m]) / static_cast<double>(p[m + s] - p[m]));
							}
							p[m] += s;
						}
					}
				}
			});
		}
	}

	cv::Mat QuantileNUC::transform(const cv::Mat& frame, std::size_t /*index*/) const {
		const int N = params_.num_quantiles;
		const int M = N + 1;
		const float invN = 1.0f / N;

		cv::Mat frame32;
		frame.convertTo(frame32, CV_32F);
		cv::Mat out(frame.size(), CV_32F);

		for (int r = 0; r < frame32.rows; ++r) {
			const float* src = frame32.ptr<float>(r);
			float* dst = out.ptr<float>(r);

			for (int c = 0; c < frame32.cols; ++c) {
				const float* q = &q_[(static_cast<std::size_t>(r) * W_ + c) * M];
				const float x = src[c];

				if (x <= q[0]) { dst[c] = 0.f; continue; }
				if (x >= q[N]) { dst[c] = 1.f; continue; }

				// сегмент i: q[i] <= x < q[i+1]
				int i = static_cast<int>(std::upper_bound(q + 1, q + N, x) - (q + 1));

				const float dq = q[i + 1] - q[i];
				const float frac = dq > 1e-6f ? (x - q[i]) / dq : 0.5f;

				dst[c] = (i + frac) * invN;   // f(x) = i/N + (1/N)(x-q_i)/(q_{i+1}-q_i)
			}
		}
		return out;
	}

}