#include "filters/column_hist_nuc.hpp"
#include "core/frame_source.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace irfilter {

	namespace {

		inline int binOf(float v, float scale, int B) {
			const int b = static_cast<int>(v * scale);
			return b < 0 ? 0 : (b >= B ? B - 1 : b);
		}

		// 1. Кумулятивные гистограммы столбцов: CV_32F [W x B]
		cv::Mat computeColumnCDFs(const FrameSource& source, int F, int W, int H, int B, float scale)
		{
			cv::Mat counts(W, B, CV_32S, cv::Scalar(0));

			for (int f = 0; f < F; ++f) {
				cv::Mat fr;
				source.at(static_cast<std::size_t>(f)).convertTo(fr, CV_32F);

				cv::parallel_for_(cv::Range(0, W), [&](const cv::Range& rg) {
					for (int y = 0; y < H; ++y) {
						const float* row = fr.ptr<float>(y);
						for (int x = rg.start; x < rg.end; ++x)
							++counts.ptr<int>(x)[binOf(row[x], scale, B)];
					}
				});
			}

			cv::Mat cdfs(W, B, CV_32F);
			const double inv = 1.0 / (static_cast<double>(H) * F);
			for (int x = 0; x < W; ++x) {
				const int* c = counts.ptr<int>(x);
				float* p = cdfs.ptr<float>(x);
				double acc = 0.0;
				for (int b = 0; b < B; ++b) {
					acc += c[b];
					p[b] = static_cast<float>(acc * inv);
				}
				p[B - 1] = 1.0f;
			}
			return cdfs;
		}

		// 2. Обратные CDF: invs[x][k] = значение (центр бина), где H_x впервые >= u_k
		cv::Mat computeInverseCDFs(const cv::Mat& cdfs, int numQuant, double maxValue)
		{
			const int W = cdfs.rows;
			const int n = cdfs.cols;
			const float binW = static_cast<float>(maxValue / n);

			cv::Mat invs(W, numQuant, CV_32F);

			cv::parallel_for_(cv::Range(0, W), [&](const cv::Range& rg) {
				for (int x = rg.start; x < rg.end; ++x) {
					const float* h = cdfs.ptr<float>(x);
					float* out = invs.ptr<float>(x);

					int j = 0;
					for (int k = 0; k < numQuant; ++k) {
						const float u = (k + 0.5f) / numQuant;
						while (j < n - 1 && h[j] < u) ++j;
						out[k] = (j + 0.5f) * binW;
					}
				}
			});
			return invs;
		}

		// 3. Целевая обратная CDF: гауссово среднее H^-1 соседей
		cv::Mat computeTargetInverseCDFs(const cv::Mat& invs, int radius, double sigma, bool includeSelf)
		{
			const int W = invs.rows;
			const int cols = invs.cols;
			cv::Mat target = invs.clone();
			if (W < 2) return target;

			std::vector<float> weights(2 * radius + 1);
			for (int d = -radius; d <= radius; ++d)
				weights[d + radius] = static_cast<float>(std::exp(-0.5 * d * d / (sigma * sigma)));

			cv::parallel_for_(cv::Range(0, W), [&](const cv::Range& rg) {
				for (int x = rg.start; x < rg.end; ++x) {
					float* out = target.ptr<float>(x);
					std::fill(out, out + cols, 0.0f);

					float wSum = 0.0f;
					for (int d = -radius; d <= radius; ++d) {
						const int xn = x + d;
						if ((d == 0 && !includeSelf) || xn < 0 || xn >= W) continue;

						const float w = weights[d + radius];
						const float* in = invs.ptr<float>(xn);
						for (int k = 0; k < cols; ++k) out[k] += w * in[k];
						wSum += w;
					}

					const float invW = 1.0f / wSum;
					for (int k = 0; k < cols; ++k) out[k] *= invW;
				}
			});
			return target;
		}

	} // namespace

	ColumnHistNUC::ColumnHistNUC(const FrameSource& source) : ColumnHistNUC(source, Params{}) {}

	ColumnHistNUC::ColumnHistNUC(const FrameSource& source, Params params)
		: FrameTransformer(source), params_(params) {
		if (params_.num_bins < 2)    throw std::runtime_error("ColumnHistNUC: num_bins must be >= 2");
		if (params_.max_value <= 0.) throw std::runtime_error("ColumnHistNUC: max_value must be > 0");
		if (params_.num_quant < 2)   throw std::runtime_error("ColumnHistNUC: num_quant must be >= 2");
		if (params_.radius < 1)      throw std::runtime_error("ColumnHistNUC: radius must be >= 1");
		if (params_.sigma <= 0.0)    throw std::runtime_error("ColumnHistNUC: sigma must be > 0");
		if (source.size() < 1)       throw std::runtime_error("ColumnHistNUC: need at least 1 frame");
		initialize();
	}

	void ColumnHistNUC::initialize() {
		const cv::Mat first = source().at(0);
		if (first.channels() != 1) throw std::runtime_error("ColumnHistNUC::initialize: expected single-channel frames");

		W_ = first.cols; H_ = first.rows;
		if (W_ < 2) throw std::runtime_error("ColumnHistNUC::initialize: need at least 2 columns");

		const int B = params_.num_bins;
		const int Q = params_.num_quant;
		bin_scale_ = static_cast<float>(B / params_.max_value);

		int F = static_cast<int>(size());
		if (params_.max_frames > 0) F = std::min(F, params_.max_frames);

		cv::Mat cdfs      = computeColumnCDFs(source(), F, W_, H_, B, bin_scale_);
		cv::Mat invs      = computeInverseCDFs(cdfs, Q, params_.max_value);
		cv::Mat targetInv = computeTargetInverseCDFs(invs, params_.radius, params_.sigma, params_.include_self);

		// 4. LUT: v' = H^-1_target( H_src(v) ), u = середина ступеньки CDF
		lut_.assign(static_cast<std::size_t>(W_) * B, 0.f);

		cv::parallel_for_(cv::Range(0, W_), [&](const cv::Range& rg) {
			for (int x = rg.start; x < rg.end; ++x) {
				const float* h    = cdfs.ptr<float>(x);
				const float* tinv = targetInv.ptr<float>(x);
				float* lut = &lut_[static_cast<std::size_t>(x) * B];

				for (int b = 0; b < B; ++b) {
					const float lo = (b > 0) ? h[b - 1] : 0.0f;
					const float u  = 0.5f * (lo + h[b]);
					const int k    = std::min(static_cast<int>(u * Q), Q - 1);
					lut[b] = tinv[k];
				}
			}
		});
	}

	cv::Mat ColumnHistNUC::transform(const cv::Mat& frame, std::size_t /*index*/) const {
		if (frame.cols != W_ || frame.rows != H_)
			throw std::runtime_error("ColumnHistNUC::transform: frame size mismatch");

		const int B = params_.num_bins;

		cv::Mat f32;
		frame.convertTo(f32, CV_32F);
		cv::Mat out(frame.size(), CV_32F);

		cv::parallel_for_(cv::Range(0, H_), [&](const cv::Range& rg) {
			for (int r = rg.start; r < rg.end; ++r) {
				const float* src = f32.ptr<float>(r);
				float* dst = out.ptr<float>(r);
				for (int c = 0; c < W_; ++c)
					dst[c] = lut_[static_cast<std::size_t>(c) * B + binOf(src[c], bin_scale_, B)];
			}
		});

		// cv::Mat filtred;

		// cv::GaussianBlur(out, filtred, cv::Size(5, 5), 1.5);

		return out;
	}

}