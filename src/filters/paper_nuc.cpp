#include "filters/paper_nuc.hpp"
#include "core/frame_source.hpp"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace irfilter {

	PaperNUC::PaperNUC(const FrameSource& source) : PaperNUC(source, Params{}) {}

	PaperNUC::PaperNUC(const FrameSource& source, Params params) : FrameTransformer(source), params_(params) {
		if (source.size() < 2) throw std::runtime_error("PaperNUC: need at least 2 frames (scene-based NUC)");
		initialize();
	}

	cv::Mat PaperNUC::ExtractXOY(int row) {
		const std::size_t total = size();
		if (total < 2) throw std::runtime_error("PaperNUC::ExtractXOY: need at least 2 frames");

		int requested = params_.frame_buff_size;
		if (requested <= 0) requested = static_cast<int>(total);

		const int F = std::min<int>(requested, static_cast<int>(total));
		if (F < 2) throw std::runtime_error("PaperNUC::ExtractXOY: frame buffer is too small");
		
		cv::Mat first = source().at(0);
		if (first.channels() != 1) throw std::runtime_error("PaperNUC::ExtractXOY: expected single-channel frames");

		const int W = first.cols;
		cv::Mat surface(W, F, CV_32F);

		for (int f=0; f<F; ++f) {
			cv::Mat frame = source().at(static_cast<std::size_t>(f));
			cv::Mat frame32;
			frame.convertTo(frame32, CV_32F);

			const float* src = frame32.ptr<float>(row);
			for (int w = 0; w < W; ++w) {
				surface.at<float>(w, f) = src[w];
			}
		}

		return surface;
	}

	void PaperNUC::SortSurfaceByRows(cv::Mat& surface) {
		if (surface.empty()) {
			return;
		}

		if (surface.type() != CV_32F) {
			surface.convertTo(surface, CV_32F);
		}

		for (int r = 0; r < surface.rows; ++r) {
			cv::Mat row = surface.row(r);
			std::sort(row.begin<float>(), row.end<float>());
		}
	}

	void PaperNUC::initialize() {
		const cv::Mat& first = source().at(0);
		if (first.channels() != 1) throw std::runtime_error("PaperNUC::initialize: expected single-channel frames");

		const int W = first.cols, H = first.rows;
		const int L = params_.block_size_L;
		if (L <= 0) throw std::runtime_error("PaperNUC::initialize: block_size_L must be > 0");

		int requested = params_.frame_buff_size;
		if (requested <= 0) requested = static_cast<int>(size());
		const int F = std::min(requested, static_cast<int>(size()));
		if (F < 2) throw std::runtime_error("PaperNUC::initialize: frame buffer is too small");

		const int HW = H*W;
		const int num_blocks = (F+L-1)/L;

		cv::Mat surface_T(F, HW, CV_32F);
		for (int f=0; f<F; ++f) {
			cv::Mat f32;
			source().at(f).convertTo(f32, CV_32F);
			std::memcpy(surface_T.ptr<float>(f), f32.ptr<float>(), HW*sizeof(float));
		}

		cv::Mat surface;
		cv::transpose(surface_T, surface);

		cv::parallel_for_(cv::Range(0, HW), [&](const cv::Range& rg) {
			for (int i = rg.start; i<rg.end; ++i) {
				float* row = surface.ptr<float>(i);
				std::sort(row, row + F);
			}
		});

		std::vector<cv::Mat> Q(num_blocks), T(num_blocks);
		for (int b=0; b<num_blocks; ++b) {
			const int col0  = b*L;
			const int width = std::min(L, F - col0);

			cv::Mat blk = surface.colRange(col0, col0+width);
			cv::Mat qflat;
			cv::reduce(blk, qflat, /*dim=*/1, cv::REDUCE_AVG);
			Q[b] = qflat.reshape(1, H);

			cv::Mat tflat;
			cv::reduce(Q[b], tflat, /*dim=*/1, cv::REDUCE_AVG);
			T[b] = tflat;
		}

		coeffs_.assign(H, std::vector<std::vector<Segment>>(W));

		for (int r=0; r<H; ++r) {
			std::vector<int> bounds;
			bounds.reserve(num_blocks);
			bounds.push_back(0);
			for (int n = 0; n + 2 < num_blocks; ++n) {
				const float Tn  = T[n  ].at<float>(r);
				const float Tn1 = T[n+1].at<float>(r);
				const float Tn2 = T[n+2].at<float>(r);
				if (std::abs((Tn-Tn1)-(Tn1-Tn2))>=params_.segment_threshold) {
					bounds.push_back(n+1);
				}
			}
			bounds.push_back(num_blocks-1);

			const int num_segs = static_cast<int>(bounds.size()) - 1;

			for (int w=0; w<W; ++w) {
				auto& segs = coeffs_[r][w];
				segs.reserve(num_segs);

				for (int s=0; s<num_segs; ++s) {
					const int start = bounds[s];
					const int end   = bounds[s + 1];

					const float Q1 = Q[start].at<float>(r, w);
					const float Q2 = Q[end  ].at<float>(r, w);
					const float T1 = T[start].at<float>(r);
					const float T2 = T[end  ].at<float>(r);

					const float dQ = Q2 - Q1;
					const float dT = T2 - T1;

					Segment seg;
					if (std::abs(dQ) > 1e-6f) {
						seg.k = dT / dQ;
						seg.b = T1 - seg.k * Q1;
					} else {
						seg.k = 1.0f;
						seg.b = 0.0f;
					}

					seg.value_min = std::min(Q1, Q2);
					seg.value_max = std::max(Q1, Q2);
					
					segs.push_back(seg);
				}
			}
		}
	}

	cv::Mat PaperNUC::transform(const cv::Mat& frame, std::size_t index) const {
		cv::Mat out(frame.size(), CV_32F);
		cv::Mat frame32;
		frame.convertTo(frame32, CV_32F);

		for (int r=0; r<frame32.rows; ++r) {
			const float* src = frame32.ptr<float>(r);
			float* dst = out.ptr<float>(r);

			for (int w=0; w<frame32.cols; ++w) {
				float val = src[w];
				const auto& segs = coeffs_[r][w];

				if (segs.empty()) {
					dst[w] = val;
					continue;
				}

				bool found = false;
				for (const auto& seg : segs) {
					if (val>=seg.value_min && val<=seg.value_max) {
						dst[w] = static_cast<float>(seg.k * val + seg.b);
						found = true;
						break;
					}
				}

				if (!found) {
					if (val < segs.front().value_min) {
						dst[w] = static_cast<float>(segs.front().k * val + segs.front().b);
					} else {
						dst[w] = static_cast<float>(segs.back().k * val + segs.back().b);
					}
				}
			}
		}

		return out;
	}

}
