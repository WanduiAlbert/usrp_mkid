
#include <iostream>
#include <vector>
#include <boost/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <time.h>
#include <assert.h>
#include <stdio.h>

using namespace std;

double timespec_diff(struct timespec *start, struct timespec *stop)
{
	double x;
	struct timespec result;
	if ((stop->tv_nsec - start->tv_nsec) < 0) {
		result.tv_sec = stop->tv_sec - start->tv_sec - 1;
		result.tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
	} else {
		result.tv_sec = stop->tv_sec - start->tv_sec;
		result.tv_nsec = stop->tv_nsec - start->tv_nsec;
	}
	x = result.tv_sec*1e6 + result.tv_nsec*1e-3;
	return x;
}

int nmc = 1010;
int nt = 16384;

void writethreadfunc(boost::lockfree::queue<vector<short> *> *message_queue, FILE *fout, int *stop_write)
{
	vector<short> *v;
	while((*stop_write == false) || message_queue->pop(v)) {
		fwrite(&(*v)[0],sizeof(short),v->size(),fout);
		delete v;
	}
}

int main(void)
{
	clockid_t clkid = CLOCK_MONOTONIC;

	//int ret;
	//ret = clock_getres(clkid,&res);
	//assert(ret == 0);
	//cout << "clock res: " << res.tv_sec << " " << res.tv_nsec << endl;

	struct timespec t0;
	struct timespec t1;

	FILE *fout = fopen("/home/lebicep_admin/data/dump.bin","w");
	assert(fout != NULL);

	vector<double> ts(nmc);

	boost::lockfree::queue<vector<short> *> message_queue(0);

	boost::thread_group writethread;
	int stop_write = false;
	writethread.create_thread(boost::bind(&writethreadfunc,&message_queue,fout,&stop_write));
	//short *v0 = (short*)malloc(sizeof(short)*nt);
	//assert(v0 != NULL);

	for(int i=0;i<nmc;i++) {
		clock_gettime(clkid,&t0);

		//short *v = (short*)malloc(sizeof(short)*nt);
		vector<short> *v = new vector<short>(nt);
		assert(v != NULL);
		message_queue.push(v);
		//fwrite(v0,sizeof(short),nt,fout);

		clock_gettime(clkid,&t1);
		ts[i] = timespec_diff(&t0,&t1);
	}
	stop_write = true;
	writethread.join_all();
	for(int i=0;i<nmc;i++) {
		cout << ts[i] << endl;
	}
	fclose(fout);

	return 0;
}
