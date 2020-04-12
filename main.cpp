#include "mainwindow.h"

#include <QApplication>

#ifdef CUDA
#include <cuda_runtime.h>
#endif

int main(int argc, char *argv[])
{
#ifdef CUDA
    int *deviceInit;
    cudaMalloc(&deviceInit, 0 * sizeof(int));
#endif

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
