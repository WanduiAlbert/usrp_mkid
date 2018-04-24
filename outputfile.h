#pragma once

#include <vector>
#include <memory>
#include "types.h"

class OutputFile
{
	public:
		typedef std::shared_ptr<OutputFile> sptr;
		std::string fn;
		FILE *file;
		size_t nchan;
		~OutputFile();
		OutputFile(std::string _fn,size_t _nchan);
		void write(std::vector<complex64> &buf);
};
