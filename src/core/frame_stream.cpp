#include "core/frame_stream.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace irfilter {

	namespace {

		std::string lowerExtension(const std::string& path) {
			std::filesystem::path p(path);
			std::string ext = p.extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(),
					[](unsigned char c){ return static_cast<char>(std::tolower(c)); });
			return ext;
		}

		template <class V>
			bool contains(const V& v, const std::string& s) {
				return std::find(v.begin(), v.end(), s) != v.end();
			}

		cv::Mat toFloatNormalized(const cv::Mat& src) {
			if (src.depth() == CV_32F) {
				return src;   // уже в нужном формате
			}
			double scale = 1.0;
			switch (src.depth()) {
				case CV_8U:  scale = 1.0 / 255.0;   break;
				case CV_16U: scale = 1.0 / 65535.0; break;
				case CV_8S:  scale = 1.0 / 127.0;   break;
				case CV_16S: scale = 1.0 / 32767.0; break;
				default:     scale = 1.0;           break;
			}
			cv::Mat f;
			src.convertTo(f, CV_32F, scale);
			return f;
		}

	}

	bool FrameStream::isImageExtension(const std::string& path) {
		static const std::vector<std::string> exts = {
			".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib",
			".tif", ".tiff", ".webp", ".pbm", ".pgm", ".ppm",
			".pnm", ".sr", ".ras", ".exr", ".hdr", ".pic", ".jp2"
		};
		return contains(exts, lowerExtension(path));
	}

	bool FrameStream::isVideoExtension(const std::string& path) {
		static const std::vector<std::string> exts = {
			".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
			".mpg", ".mpeg", ".m4v", ".3gp", ".ts", ".mts", ".m2ts", ".ogv"
		};
		return contains(exts, lowerExtension(path));
	}

	bool FrameStream::loadFromFile(const std::string& path) {
		if (isImageExtension(path)) return loadFromImage(path);
		if (isVideoExtension(path)) return loadFromVideo(path);
		return loadFromImage(path) || loadFromVideo(path);
	}

	bool FrameStream::loadFromImage(const std::string& path) {
		cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
		if (img.empty()) return false;

		clear();
		frames_.push_back(toFloatNormalized(img));   // ← ключевая правка
		sourcePath_ = path;
		fps_        = 0.0;
		return true;
	}

	bool FrameStream::loadFromVideo(const std::string& path) {
		cv::VideoCapture cap(path);
		if (!cap.isOpened()) return false;

		const double srcFps = cap.get(cv::CAP_PROP_FPS);

		Container loaded;
		cv::Mat  frame;
		while (cap.read(frame)) {
			if (frame.empty()) break;
			loaded.push_back(toFloatNormalized(frame));
		}
		cap.release();

		if (loaded.empty()) return false;

		frames_     = std::move(loaded);
		sourcePath_ = path;
		fps_        = srcFps;
		return true;
	}

	void FrameStream::addFrame(const Frame& frame) {
		frames_.push_back(toFloatNormalized(frame));
	}
	void FrameStream::addFrame(Frame&& frame) {
		frames_.push_back(toFloatNormalized(frame));
	}

	void FrameStream::clear() noexcept {
		frames_.clear();
		sourcePath_.clear();
		fps_ = 0.0;
	}

}
