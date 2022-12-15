#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <QString>
#include <QUrl>
#include <QDesktopServices>

#include "..\..\Other\Utility.h"

#define gpuErrchk(ans) CUDAUtility::gpuAssert((ans), __FILE__, __LINE__)

#define cudaErrStr(errCode) CUDAUtility::createCUDAErrStr((errCode), __FILE__, __LINE__)

namespace CUDAUtility
{
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort)
                exit(code);
        }
    }

    inline QString createCUDAErrStr(const cudaError_t code, const char *file, const int line)
    {
        return QString("CUDA Error (%1): %2\n%3 line %4").arg(static_cast<int>(code)).arg(cudaGetErrorString(code), file).arg(line);
    }

    enum class cudaErrorType { SUCCESS, ERR_EXPECTED, ERR_UNEXPECTED };

    inline cudaErrorType CUDAErrMessageBox(QWidget *parent, const QString &msg, const cudaError code, const std::initializer_list<cudaError> &expectedCodes, const char *file, const int line)
    {
        cudaErrorType result = cudaErrorType::SUCCESS;
        if (code != cudaSuccess)
        {
            QString title("CUDA Error");
            QString errorMessage(msg);
            errorMessage.append("\n");
            QMessageBox::StandardButtons buttons = QMessageBox::StandardButton::Ok;

            //If the error isn't one of the expected ones then something has gone extra wrong...
            if (std::find(expectedCodes.begin(), expectedCodes.end(), code) == expectedCodes.end())
            {
                title.prepend("Unexpected ");
                errorMessage.append("Please report this with the logs at: https://github.com/MorganGrundy/MosaicMagnifique/issues \n");
                buttons |= QMessageBox::Open;
                result = cudaErrorType::ERR_UNEXPECTED;
            }
            else
                result = cudaErrorType::ERR_EXPECTED;

            errorMessage.append(createCUDAErrStr(code, file, line));

            if (MessageBox::critical(parent, title, errorMessage, buttons) == QMessageBox::Open)
                QDesktopServices::openUrl(QUrl("https://github.com/MorganGrundy/MosaicMagnifique/issues"));
        }

        return result;
    }
};