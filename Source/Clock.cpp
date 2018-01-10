#include "Clock.h"

Clock::Clock()
{
	time = 0;
}

Clock::~Clock()
{
}

void Clock::Start()
{
    begin = std::chrono::high_resolution_clock::now();
}

void Clock::Stop()
{
	//std::chrono::high_resolution_clock::time_point tmp = std::chrono::high_resolution_clock::now();
	time += std::chrono::duration_cast<std::chrono::duration<unsigned long long,std::micro>>(std::chrono::high_resolution_clock::now() - begin).count();
	//std::cout << std::chrono::duration_cast<std::chrono::duration<unsigned long long, std::micro>>(tmp - begin).count() << std::endl;
	begin = std::chrono::high_resolution_clock::now();
}

void Clock::Reset()
{
	time = 0;
}

unsigned long long Clock::ElapsedMilliseconds()
{
    return time/1000;
}

double Clock::ElapsedSeconds()
{
    return time/(double)1000000;
}