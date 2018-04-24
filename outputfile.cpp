
#include "outputfile.h"

#include <string>
#include <queue>

using namespace std;

OutputFile::~OutputFile() {
	fclose(file);
}

OutputFile::OutputFile(string _fn,size_t _nchan) : fn(_fn),nchan(_nchan) {
	file = fopen(fn.c_str(),"w");
	fwrite(&nchan,sizeof(nchan),1,file);
}

void OutputFile::write(vector<complex64> &buf) {
	fwrite(&buf[0],sizeof(complex64),buf.size(),file);
}

