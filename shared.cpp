#ifndef SHARED_CPP_
#define SHARED_CPP_

#include <iostream>

#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>

using namespace std;

void progressBar(int cur, int max, int width)
{
    int bar_width = width - 8;
    int progress = (cur * bar_width) / max;
    cout << "[";
    for (int i = 0; i < bar_width; i++)
        if (i < progress)
            cout << "=";
        else
            cout << " ";
    cout << "] " << (cur * 100) / max << "% \r";
    cout.flush();
}

#endif //SHARED_CPP_
