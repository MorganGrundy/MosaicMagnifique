#include <iostream>

#include "Benchmark_Generator.h"

int main(int argc, char *argv[])
{
	if (argc > 1)
	{
		std::cout << "Mosaic Magnifique CLI\n";

		if (argc < 3)
		{
			std::cout << "Must provide atleast mil and main image file\n";
			return 0;
		}

		QString libFile(argv[1]);
		std::string mainImageFile(argv[2]);
		QString cellShapeFile;
		bool useCUDA = true;
		ColourDifference::Type colourDiff = ColourDifference::Type::CIE76;
		int detail = 100;
		size_t sizeSteps = 0;
		size_t cellSize = 128;
		double mainImageSize = 1.0;
		int repeatRange = 0, repeatAddition = 0;
		ColourScheme::Type colourScheme = ColourScheme::Type::NONE;

		if (argc > 3)
			cellShapeFile = argv[3];

		if (argc > 4)
			useCUDA = (argv[4] == "1") ? true : false;

		if (argc > 5)
			colourDiff = static_cast<ColourDifference::Type>(std::stoi(argv[5]));

		if (argc > 6)
			detail = std::stoi(argv[6]);

		if (argc > 7)
			sizeSteps = std::stoull(argv[7]);

		if (argc > 8)
			cellSize = std::stoull(argv[8]);

		if (argc > 9)
			mainImageSize = std::stod(argv[9]);

		if (argc > 10)
			repeatRange = std::stoi(argv[10]);

		if (argc > 11)
			repeatAddition = std::stoi(argv[11]);

		if (argc > 12)
			colourScheme = static_cast<ColourScheme::Type>(std::stoi(argv[12]));

		auto time = GeneratePhotomosaic(libFile, mainImageFile, cellShapeFile, useCUDA, colourDiff, detail, sizeSteps, cellSize, mainImageSize, repeatRange, repeatAddition, colourScheme);
		std::cout << "Generated in " << time << "ms\n";
	}
	else
	{
		std::cout << "Mosaic Magnifique Benchmark\n";

		Benchmark_Generator();
	}
}