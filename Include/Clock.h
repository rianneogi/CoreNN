#pragma once

#include <chrono>
#include <sys/timeb.h>
#include <iostream>
#include <conio.h>

#pragma warning(disable: 4996) // Function _ftime() may be unsafe

class Clock
{
public:
	std::chrono::high_resolution_clock::time_point begin;
    //std::chrono::high_resolution_clock::time_point end;

	uint64_t time;
	//std::chrono::time_point<std::chrono::high_resolution_clock> t;

	Clock();
	~Clock();
	void Start();
	void Stop();
	void Reset();
	uint64_t ElapsedMilliseconds();
	double ElapsedSeconds();
};

inline uint64_t system_time_to_msec() {
	_timeb t;
	_ftime(&t);
	return t.time * 1000LL + t.millitm;
}

