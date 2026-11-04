#include "core/frame_stream.hpp"
#include "metrics/simple_metrics.hpp"
#include "noises/detector_nonuniformity.hpp"
#include "filters/paper_nuc.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

	cv::Mat toU8(const cv::Mat& f) {
		cv::Mat f32;
		if (f.type() != CV_32F) f.convertTo(f32, CV_32F);
		else                    f32 = f;
		cv::Mat u8;
		f32.convertTo(u8, CV_8U, 255.0);
		return u8;
	}

	bool writeVideo(const std::string& path,
			const irfilter::FrameSource& src,
			double fps = 25.0)
	{
		if (src.size() == 0) { std::cerr << "writeVideo: empty\n"; return false; }

		cv::Mat first = toU8(src.at(0));
		const bool isColor = (first.channels() == 3);

		cv::VideoWriter vw(path,
				cv::VideoWriter::fourcc('m','p','4','v'),
				fps, first.size(), isColor);

		if (!vw.isOpened()) {
			std::cerr << "writeVideo: cannot open '" << path << "'\n";
			return false;
		}

		for (std::size_t i=0; i<src.size(); ++i) {
			cv::Mat u8 = toU8(src.at(i));
			if (u8.size() != first.size()) {
				cv::resize(u8, u8, first.size());
			}
			vw.write(u8);
		}
		std::cout << "saved " << src.size() << " frames -> " << path << "\n";
		return true;
	}

} // namespace

int main(int argc, char** argv) {
	if (argc < 2) { std::cerr << "usage: ir_filter_tester <path>\n"; return 1; }

	irfilter::FrameStream stream;
	if (!stream.loadFromFile(argv[1])) { std::cerr << "cannot load\n"; return 1; }

	if (stream.size() < 5) {
		std::vector<cv::Mat> copies;
		copies.reserve(10);
		for (int k = 0; k < 10; ++k) copies.push_back(stream[0].clone());
		stream.clear();
		for (auto& f : copies) stream.addFrame(std::move(f));
	}

	for (auto& f : stream) {
		if (f.channels() == 3) {
			cv::Mat g; cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
			f = g;
		}
	}

	irfilter::PaperNUC::Params p;
	p.dump_dir = "nuc_debug";

	irfilter::DetectorNonuniformityNoise noisy(stream, {}, /*seed=*/42);
	irfilter::PaperNUC nuc(noisy, p);
	irfilter::MAE mae;
	irfilter::MSE mse;

	const double maeIn  = irfilter::evaluate(mae, stream, noisy);
	const double mseIn  = irfilter::evaluate(mse, stream, noisy);
	const double maeOut = irfilter::evaluate(mae, stream, nuc);
	const double mseOut = irfilter::evaluate(mse, stream, nuc);

	std::cout << "MAE  (noisy)  = " << maeIn  << "\n"
		<< "MSE  (noisy)  = " << mseIn  << "\n"
		<< "MAE  (NUC)    = " << maeOut << "\n"
		<< "MSE  (NUC)    = " << mseOut << "\n";

	cv::imwrite("input_0.png",     toU8(stream.at(0)));
	cv::imwrite("noisy_0.png",     toU8(noisy.at(0)));
	cv::imwrite("corrected_0.png", toU8(nuc.at(0)));

	writeVideo("original_gray.mp4", stream);
	writeVideo("noisy.mp4",         noisy);
	writeVideo("corrected.mp4",     nuc);

	return 0;
}
