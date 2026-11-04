#ifndef FRAME_STREAM_H
#define FRAME_STREAM_H

#include "core/frame_source.hpp"

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <cstddef>

namespace irfilter {

	class FrameStream : public FrameSource {
		public:
			using Frame     = cv::Mat;
			using Container = std::vector<Frame>;
			using iterator       = Container::iterator;
			using const_iterator = Container::const_iterator;

			FrameStream() = default;
			explicit FrameStream(const std::string& path) { loadFromFile(path); }

			bool loadFromFile(const std::string& path);
			bool loadFromImage(const std::string& path);
			bool loadFromVideo(const std::string& path);

			void addFrame(const Frame& frame);
			void addFrame(Frame&& frame);

			void clear() noexcept;
			bool empty() const noexcept { return frames_.empty(); }
			std::size_t size() const noexcept override { return frames_.size(); }

			Frame&       operator[](std::size_t i)        { return frames_[i]; }
			const Frame& operator[](std::size_t i) const  { return frames_[i]; }
			cv::Mat      at(std::size_t i) const override { return frames_.at(i); }
			Frame&       front()       { return frames_.front(); }
			const Frame& front() const { return frames_.front(); }
			Frame&       back()        { return frames_.back();  }
			const Frame& back()  const { return frames_.back();  }

			iterator       begin()  noexcept       { return frames_.begin(); }
			iterator       end()    noexcept       { return frames_.end();   }
			const_iterator begin()  const noexcept { return frames_.begin(); }
			const_iterator end()    const noexcept { return frames_.end();   }
			const_iterator cbegin() const noexcept { return frames_.cbegin(); }
			const_iterator cend()   const noexcept { return frames_.cend();  }

			double             fps()        const noexcept { return fps_; }
			const std::string& sourcePath() const noexcept { return sourcePath_; }
			const Container&   frames()     const noexcept { return frames_; }

		private:
			static bool isImageExtension(const std::string& path);
			static bool isVideoExtension(const std::string& path);

			Container   frames_;
			std::string sourcePath_;
			double      fps_ = 0.0;
	};

}

#endif
