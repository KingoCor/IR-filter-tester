#include "noises/detector_nonuniformity.hpp"

#include <cmath>
#include <stdexcept>

namespace irfilter {

	DetectorNonuniformityNoise::DetectorNonuniformityNoise(const FrameSource& source, std::uint64_t seed): 
		DetectorNonuniformityNoise(source, Params{}, seed) {}

	DetectorNonuniformityNoise::DetectorNonuniformityNoise(const FrameSource& source, Params params, std::uint64_t seed)
		: NoiseModel(source, seed), params_(params)
	{
		if (source.size() == 0)
			throw std::runtime_error("DetectorNonuniformityNoise: empty source");
	}

	void DetectorNonuniformityNoise::initialize(int H, int W) const {
		cv::RNG rng(seed());

		A_ = cv::Mat(H, W, CV_32F);
		B_ = cv::Mat(H, W, CV_32F);
		C_ = cv::Mat(H, W, CV_32F);
		D_ = cv::Mat(H, W, CV_32F);

		for (int y=0; y<H; ++y) {
			for (int x=0; x<W; ++x) {
				A_.at<float>(y, x) = rng.uniform(params_.A_min, params_.A_max);
				B_.at<float>(y, x) = rng.uniform(params_.B_min, params_.B_max);
				C_.at<float>(y, x) = rng.uniform(params_.C_min, params_.C_max);
				D_.at<float>(y, x) = rng.uniform(params_.D_min, params_.D_max);
			}
		}

		A_n_.resize(W); B_n_.resize(W); C_n_.resize(W); D_n_.resize(W);
		for (int i=0; i<W; ++i) {
			A_n_[i] = rng.uniform(params_.A_min, params_.A_max);
			B_n_[i] = rng.uniform(params_.B_min, params_.B_max);
			C_n_[i] = rng.uniform(params_.C_min, params_.C_max);
			D_n_[i] = rng.uniform(params_.D_min, params_.D_max);
		}
	}

	cv::Mat DetectorNonuniformityNoise::corrupt(const cv::Mat& frame, std::uint64_t /*seed*/) const {
		if (frame.channels() != 1) throw std::runtime_error("DetectorNonuniformityNoise: expect grayscale (1 channel) input");
		CV_Assert(frame.type() == CV_32FC1);

		if (!initialized_) { initialize(frame.rows, frame.cols); initialized_ = true; }

		const int H = frame.rows, W = frame.cols;

		// probe
		cv::Mat DN1(H, W, CV_32F);
		for (int y=0; y<H; ++y) {
			const float* img = frame.ptr<float>(y);
			float* out = DN1.ptr<float>(y);

			const float* A = A_.ptr<float>(y);
			const float* B = B_.ptr<float>(y);
			const float* C = C_.ptr<float>(y);
			const float* D = D_.ptr<float>(y);

			for (int x=0; x<W; ++x) {
				const float t = -B[x]*(img[x]-C[x]);
				out[x] = A[x]/(1.0f+std::exp(t))+D[x];
			}
		}

		// readout
		cv::Mat DN2(H, W, CV_32F);
		for (int y=0; y<H; ++y) {
			const float* in = DN1.ptr<float>(y);
			float* out = DN2.ptr<float>(y);

			for (int x=0; x<W; ++x) {
				const float t = -B_n_[x]*(in[x]-C_n_[x]);
				out[x] = A_n_[x]/(1.0f+std::exp(t))+D_n_[x];
			}
		}

		return DN2;
	}

}
