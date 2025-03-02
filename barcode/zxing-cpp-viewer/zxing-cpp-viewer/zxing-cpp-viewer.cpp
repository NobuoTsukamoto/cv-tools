// zxing-cpp-viewer.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include <vector>
#include <ZXing/ReadBarcode.h>
#include <ZXing/BitMatrix.h>
#include <opencv2/opencv.hpp>

#define CVUI_IMPLEMENTATION
#include "cvui.h"


void convertBinaryBitmapToMat(const ZXing::BinaryBitmap& bitmap, cv::Mat& mat)
{
	auto data = bitmap.getBitMatrix();
	if (data != nullptr)
	{
		auto heigth = mat.rows;
		auto width = mat.cols;

		for (auto y = 0; y < heigth; ++y)
		{
			for (auto x = 0; x < width; ++x)
			{
				mat.at<uchar>(y, x) = data->get(x, y) ? 0 : 255;
			}
		}
	}
}

int main()
{
	const cv::String MAIN_WINDOW = "ZXing-Cpp Viewer";
	const cv::String SETTING_WINDOW = "Settings";

	const cv::Scalar COLOR_RED(0, 0, 255);

	cv::namedWindow(MAIN_WINDOW);
	cv::namedWindow(SETTING_WINDOW);

	cvui::init(MAIN_WINDOW);
	cvui::init(SETTING_WINDOW);

	cvui::watch(SETTING_WINDOW);

	// Open camera capture.
	cv::VideoCapture cap(0);

	std::cout << "Start capture." << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;


	const std::vector<cv::String> support_barcode_string = {
		"Aztec", "Codabar", "Code39", "Code93", "Code128",
		"DataBar", "DataBarExpanded", "DataMatrix", "EAN8", "EAN13",
		"ITF", "MaxiCode", "PDF417", "QRCode", "UPCA",
		"UPCE", "MicroQRCode", "RMQRCode", "DXFilmEdge", "DataBarLimited" };
	std::vector<bool> barcode_formats(support_barcode_string.size(), true);
	//bool barcode_formats[20] = {  };

	auto linear_codes = true;
	auto matrix_codes = true;
	auto try_harder = false;
	auto try_rotate = false;
	auto try_invert = false;
	auto try_downscale = false;
	auto is_pure = false;

	auto downscale_factor = 2.0;
	auto min_line_count = 2.0;

	auto binarizer_local_average = true;
	auto binarizer_global_histogram = false;
	auto binarizer_fixed_threshold = false;

	auto is_return_errors = false;

	while (cap.isOpened())
	{
		cv::Mat frame, gray, display_input_image, display_image, input_image;

		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		int width = frame.cols;
		int height = frame.rows;

		// Draw setting panel.
		cvui::context(SETTING_WINDOW);
		cv::Mat setting_panel = cv::Mat(600, 600, CV_8UC3);
		setting_panel = cv::Scalar(49, 52, 49);

		// Barcode Formats
		cvui::text(setting_panel, 10, 10, "Barcode Formats");
		cvui::rect(setting_panel, 5, 25, 595, 150, 0xffffff);
		cvui::beginRow(setting_panel, 10, 25, -1, -1, 50);
		for (auto i = 0; i < 5; ++i)
		{
			cvui::beginColumn(setting_panel, 10 + (i * 100), 35, -1, -1, 20);
			for (auto j = 0; j < 4; ++j)
			{
				auto index = i * 4 + j;
				if (index >= support_barcode_string.size())
				{
					break;
				}
				bool checked = barcode_formats[index];
				barcode_formats[index] = cvui::checkbox(support_barcode_string[index], &checked);
			}
			cvui::endColumn();
		}
		cvui::endRow();


		cvui::beginColumn(setting_panel, 10, 210, -1, -1, 20);
		{
			// tryHarder
			cvui::checkbox("Try Harder", &try_harder);
			// tryRotate
			cvui::checkbox("Try Rotate", &try_rotate);
			// tryInvert
			cvui::checkbox("Try Invert", &try_invert);
			// tryHarder
			cvui::checkbox("Try Downscale", &try_downscale);
			// isPure
			cvui::checkbox("Is Pure", &is_pure);
			// returnErrors
			cvui::checkbox("Return Errors", &is_return_errors);
		}
		cvui::endColumn();

		// Binarizer options
		cvui::beginColumn(setting_panel, 150, 210, -1, -1, 20);
		{

			// LocalAverage
			if (cvui::checkbox("LocalAverage", &binarizer_local_average))
			{
				binarizer_global_histogram = false;
				binarizer_fixed_threshold = false;
			}
			// GlobalHistogram
			if (cvui::checkbox("GlobalHistogram", &binarizer_global_histogram))
			{
				binarizer_local_average = false;
				binarizer_fixed_threshold = false;
			}
			// FixedThreshold
			if (cvui::checkbox("Otsu + FixedThreshold", &binarizer_fixed_threshold))
			{
				binarizer_local_average = false;
				binarizer_global_histogram = false;
			}
		}
		cvui::endColumn();

		cvui::beginColumn(setting_panel, 300, 210, -1, -1, 20);
		{
			cvui::text("Downscale Factor");
			cvui::trackbar(250, &downscale_factor, (double)2, (double)4, 1, "%.1Lf", cvui::TRACKBAR_DISCRETE);

			cvui::text("Min Line Count");
			cvui::trackbar(250, &min_line_count, (double)1.0, (double)20.0, 2, "%.1Lf", cvui::TRACKBAR_DISCRETE);
		}
		cvui::endColumn();

		cvui::update(SETTING_WINDOW);
		cv::imshow(SETTING_WINDOW, setting_panel);


		cvui::context(MAIN_WINDOW);
		// Convert to grayscale.
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Set barcode formats.
		auto options = ZXing::ReaderOptions();
		auto formats = ZXing::BarcodeFormat::None;

		for (auto i = 0; i < support_barcode_string.size(); ++i)
		{
			if (barcode_formats[i])
			{
				formats = static_cast<ZXing::BarcodeFormat>(static_cast<uint64_t>(formats) | (1 << i));
			}
		}

		options.setFormats(formats);
		options.setTryHarder(try_harder);
		options.setTryRotate(try_rotate);
		options.setTryInvert(try_invert);
		options.setIsPure(is_pure);
		options.setTryDownscale(try_downscale);
		options.setReturnErrors(is_return_errors);
		if (try_downscale)
		{
			options.setDownscaleFactor((uint8_t)downscale_factor);
		}
		options.setMinLineCount((uint8_t)min_line_count);


		std::unique_ptr<ZXing::ImageView> image = nullptr;
		if (binarizer_local_average)
		{
			options.setBinarizer(ZXing::Binarizer::LocalAverage);

			input_image = gray;
			image = std::make_unique<ZXing::ImageView>(input_image.data, width, height, ZXing::ImageFormat::Lum);

			// get binary image.
			display_image = cv::Mat::zeros(gray.size(), CV_8U);
			auto binay = CreateBitmap(ZXing::Binarizer::LocalAverage, *image);

			// zxing-cpp binary to cv::Mat
			convertBinaryBitmapToMat(*binay, display_image);
			cv::cvtColor(display_image, display_input_image, cv::COLOR_GRAY2BGR);
		}
		else if (binarizer_global_histogram)
		{
			options.setBinarizer(ZXing::Binarizer::GlobalHistogram);

			input_image = gray;
			image = std::make_unique<ZXing::ImageView>(input_image.data, width, height, ZXing::ImageFormat::Lum);

			// get binary image.
			display_image = cv::Mat::zeros(gray.size(), CV_8U);
			auto binay = CreateBitmap(ZXing::Binarizer::GlobalHistogram, *image);

			// zxing-cpp binary to cv::Mat
			convertBinaryBitmapToMat(*binay, display_image);
			cv::cvtColor(display_image, display_input_image, cv::COLOR_GRAY2BGR);
		}
		else
		{
			options.setBinarizer(ZXing::Binarizer::FixedThreshold);

			cv::threshold(gray, input_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

			// cv::adaptiveThreshold(gray, input_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 4);

			image = std::make_unique<ZXing::ImageView>(input_image.data, width, height, ZXing::ImageFormat::Lum);

			cv::cvtColor(input_image, display_input_image, cv::COLOR_GRAY2BGR);
		}


		auto barcodes = ZXing::ReadBarcodes(*image, options);


		for (const auto& barcode : barcodes)
		{
			cv::String text = ZXing::ToString(barcode.format()) + ": " + barcode.text();

			std::cout << text << "\n";

			// Display color image.
			auto pos = barcode.position();
			auto zx2cv = [](ZXing::PointI p) { return cv::Point(p.x, p.y); };
			auto contour = std::vector<cv::Point>{ zx2cv(pos[0]), zx2cv(pos[1]), zx2cv(pos[2]), zx2cv(pos[3]) };
			const auto* pts = contour.data();
			int npts = contour.size();

			cv::polylines(frame, &pts, &npts, 1, true, COLOR_RED);
			cv::putText(frame, text, zx2cv(pos[3]) + cv::Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, COLOR_RED);

			cv::polylines(display_input_image, &pts, &npts, 1, true, COLOR_RED);
			cv::putText(display_input_image, text, zx2cv(pos[3]) + cv::Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 0.5, COLOR_RED);
		}


		// Display image.
		cv::hconcat(frame, display_input_image, display_image);
		cvui::update();
		cvui::imshow(MAIN_WINDOW, display_image);

		const int key = cv::waitKey(1);
		if (key == 27 || key == 'q')
		{
			break;
		}

	}
}
