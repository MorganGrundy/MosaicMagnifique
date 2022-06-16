#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda.h>

__global__ void euclidNoPow(float *colour1, float *colour2, double *result, int t_size)
{
    for (int i = 0; i < t_size; ++i)
    {
        result[i] = sqrt(((double)colour1[i * 3] - colour2[i * 3]) * ((double)colour1[i * 3] - colour2[i * 3]) +
            ((double)colour1[i * 3 + 1] - colour2[i * 3 + 1]) * ((double)colour1[i * 3 + 1] - colour2[i * 3 + 1]) +
            ((double)colour1[i * 3 + 2] - colour2[i * 3 + 2]) * ((double)colour1[i * 3 + 2] - colour2[i * 3 + 2]));
    }
}

__global__ void euclidPow(float *colour1, float *colour2, double *result, int t_size)
{
    for (int i = 0; i < t_size; ++i)
    {
        result[i] = sqrt(pow((double)colour1[i * 3] - colour2[i * 3], 2.0) +
            pow((double)colour1[i * 3 + 1] - colour2[i * 3 + 1], 2.0) +
            pow((double)colour1[i * 3 + 2] - colour2[i * 3 + 2], 2.0));
    }
}

void euclidNoPowWrapper(float *colour1, float *colour2, double *result, int t_size)
{
    euclidNoPow<<<1, 1>>>(colour1, colour2, result, t_size);
}

void euclidPowWrapper(float *colour1, float *colour2, double *result, int t_size)
{
    euclidPow<<<1, 1>>>(colour1, colour2, result, t_size);
}

//Converts degrees to radians
__device__ inline constexpr double degToRadKernel(const double deg)
{
    return (deg * CUDART_PI) / 180.0;
}

__global__ void CIEDE2000Old(float *colour1, float *colour2, double *result, int t_size)
{
    for (int i = 0; i < t_size; ++i)
    {
        constexpr double deg360InRad = degToRadKernel(360.0);
        constexpr double deg180InRad = degToRadKernel(180.0);
        constexpr double pow25To7 = 6103515625.0; //pow(25, 7)

        const double C1 = sqrt((double)(colour1[i * 3 + 1] * colour1[i * 3 + 1]) + (colour1[i * 3 + 2] * colour1[i * 3 + 2]));
        const double C2 = sqrt((double)(colour2[i * 3 + 1] * colour2[i * 3 + 1]) + (colour2[i * 3 + 2] * colour2[i * 3 + 2]));
        const double barC = (C1 + C2) / 2.0;

        const double G = 0.5 * (1 - sqrt(pow(barC, (double)7.0) / (pow(barC, (double)7.0) + pow25To7)));

        const double a1Prime = (1.0 + G) * colour1[i * 3 + 1];
        const double a2Prime = (1.0 + G) * colour2[i * 3 + 1];

        const double CPrime1 = sqrt((a1Prime * a1Prime) + (colour1[i * 3 + 2] * colour1[i * 3 + 2]));
        const double CPrime2 = sqrt((a2Prime * a2Prime) + (colour2[i * 3 + 2] * colour2[i * 3 + 2]));

        double hPrime1;
        if (colour1[i * 3 + 2] == 0.0f && a1Prime == 0.0)
            hPrime1 = 0.0;
        else
        {
            hPrime1 = atan2((double)colour1[i * 3 + 2], a1Prime);
            //This must be converted to a hue angle in degrees between 0 and 360 by
            //addition of 2 pi to negative hue angles.
            if (hPrime1 < 0.0)
                hPrime1 += deg360InRad;
        }

        double hPrime2;
        if (colour2[i * 3 + 2] == 0.0f && a2Prime == 0.0)
            hPrime2 = 0.0;
        else
        {
            hPrime2 = atan2((double)colour2[i * 3 + 2], a2Prime);
            //This must be converted to a hue angle in degrees between 0 and 360 by
            //addition of 2pi to negative hue angles.
            if (hPrime2 < 0.0)
                hPrime2 += deg360InRad;
        }

        const double deltaLPrime = colour2[i * 3] - colour1[i * 3];
        const double deltaCPrime = CPrime2 - CPrime1;

        double deltahPrime;
        const double CPrimeProduct = CPrime1 * CPrime2;
        if (CPrimeProduct == 0.0)
            deltahPrime = 0.0;
        else
        {
            //Avoid the fabs() call
            deltahPrime = hPrime2 - hPrime1;
            if (deltahPrime < -deg180InRad)
                deltahPrime += deg360InRad;
            else if (deltahPrime > deg180InRad)
                deltahPrime -= deg360InRad;
        }

        const double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

        const double barLPrime = (colour1[i * 3] + colour2[i * 3]) / 2.0;
        const double barCPrime = (CPrime1 + CPrime2) / 2.0;

        double barhPrime;
        const double hPrimeSum = hPrime1 + hPrime2;
        if (CPrime1 * CPrime2 == 0.0)
            barhPrime = hPrimeSum;
        else
        {
            if (fabs(hPrime1 - hPrime2) <= deg180InRad)
                barhPrime = hPrimeSum / 2.0;
            else
            {
                if (hPrimeSum < deg360InRad)
                    barhPrime = (hPrimeSum + deg360InRad) / 2.0;
                else
                    barhPrime = (hPrimeSum - deg360InRad) / 2.0;
            }
        }

        const double T = 1.0 - (0.17 * cos(barhPrime - degToRadKernel(30.0))) +
            (0.24 * cos(2.0 * barhPrime)) +
            (0.32 * cos((3.0 * barhPrime) + degToRadKernel(6.0))) -
            (0.20 * cos((4.0 * barhPrime) - degToRadKernel(63.0)));

        const double deltaTheta = degToRadKernel(30.0) *
            exp(-pow((barhPrime - degToRadKernel(275.0)) / degToRadKernel(25.0), 2.0));

        const double R_C = 2.0 * sqrt(pow(barCPrime, (double)7.0) /
            (pow(barCPrime, (double)7.0) + pow25To7));

        const double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, (double)2.0)) /
            sqrt(20 + pow(barLPrime - 50.0, (double)2.0)));
        const double S_C = 1.0 + (0.045 * barCPrime);
        const double S_H = 1.0 + (0.015 * barCPrime * T);

        const double R_T = (-sin(2.0 * deltaTheta)) * R_C;

        //constexpr double k_L = 1.0, k_C = 1.0, k_H = 1.0;
        result[i] = sqrt(pow(deltaLPrime / (/*k_L * */S_L), 2.0) +
                    pow(deltaCPrime / (/*k_C * */ S_C), 2.0) +
                    pow(deltaHPrime / (/*k_H * */ S_H), 2.0) +
                (R_T * (deltaCPrime / (/*k_C * */ S_C)) *
                       (deltaHPrime / (/*k_H * */ S_H))));
    }
}

__global__ void CIEDE2000New(float *colour1, float *colour2, double *result, int t_size)
{
    for (int i = 0; i < t_size; ++i)
    {
        constexpr double deg360InRad = degToRadKernel(360.0);
        constexpr double deg180InRad = degToRadKernel(180.0);
        constexpr double pow25To7 = 6103515625.0; //pow(25, 7)

        const double C1 = sqrt((double)(colour1[i*3 + 1] * colour1[i*3 + 1]) + (colour1[i*3 + 2] * colour1[i*3 + 2]));
        const double C2 = sqrt((double)(colour2[i*3 + 1] * colour2[i*3 + 1]) + (colour2[i*3 + 2] * colour2[i*3 + 2]));
        const double barC = (C1 + C2) / 2.0;

        const double barCPow7 = barC * barC * barC * barC * barC * barC * barC;
        const double G = 0.5 * (1.0 - sqrt(barCPow7 / (barCPow7 + pow25To7)));

        const double a1Prime = (1.0 + G) * colour1[i*3 + 1];
        const double a2Prime = (1.0 + G) * colour2[i*3 + 1];

        const double CPrime1 = sqrt((a1Prime * a1Prime) + (colour1[i*3 + 2] * colour1[i*3 + 2]));
        const double CPrime2 = sqrt((a2Prime * a2Prime) + (colour2[i*3 + 2] * colour2[i*3 + 2]));

        double hPrime1;
        if (colour1[i*3 + 2] == 0.0f && a1Prime == 0.0)
            hPrime1 = 0.0;
        else
        {
            hPrime1 = atan2((double)colour1[i*3 + 2], a1Prime);
            //This must be converted to a hue angle in degrees between 0 and 360 by
            //addition of 2 pi to negative hue angles.
            if (hPrime1 < 0.0)
                hPrime1 += deg360InRad;
        }

        double hPrime2;
        if (colour2[i*3 + 2] == 0.0f && a2Prime == 0.0)
            hPrime2 = 0.0;
        else
        {
            hPrime2 = atan2((double)colour2[i*3 + 2], a2Prime);
            //This must be converted to a hue angle in degrees between 0 and 360 by
            //addition of 2pi to negative hue angles.
            if (hPrime2 < 0.0)
                hPrime2 += deg360InRad;
        }

        const double deltaLPrime = colour2[i*3] - colour1[i*3];
        const double deltaCPrime = CPrime2 - CPrime1;

        double deltahPrime;
        const double CPrimeProduct = CPrime1 * CPrime2;
        if (CPrimeProduct == 0.0)
            deltahPrime = 0.0;
        else
        {
            //Avoid the fabs() call
            deltahPrime = hPrime2 - hPrime1;
            if (deltahPrime < -deg180InRad)
                deltahPrime += deg360InRad;
            else if (deltahPrime > deg180InRad)
                deltahPrime -= deg360InRad;
        }

        const double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

        const double barLPrime = (colour1[i*3] + colour2[i*3]) / 2.0;
        const double barCPrime = (CPrime1 + CPrime2) / 2.0;

        double barhPrime;
        const double hPrimeSum = hPrime1 + hPrime2;
        if (CPrime1 * CPrime2 == 0.0)
            barhPrime = hPrimeSum;
        else
        {
            if (fabs(hPrime1 - hPrime2) <= deg180InRad)
                barhPrime = hPrimeSum / 2.0;
            else
            {
                if (hPrimeSum < deg360InRad)
                    barhPrime = (hPrimeSum + deg360InRad) / 2.0;
                else
                    barhPrime = (hPrimeSum - deg360InRad) / 2.0;
            }
        }

        const double T = 1.0 - (0.17 * cos(barhPrime - degToRadKernel(30.0))) +
            (0.24 * cos(2.0 * barhPrime)) +
            (0.32 * cos((3.0 * barhPrime) + degToRadKernel(6.0))) -
            (0.20 * cos((4.0 * barhPrime) - degToRadKernel(63.0)));

        const double deltaTheta = degToRadKernel(30.0) *
            exp(-((barhPrime - degToRadKernel(275.0)) * (barhPrime - degToRadKernel(275.0)) / (degToRadKernel(25.0) * degToRadKernel(25.0))));
        //((a-b)/c)^2 = ((a-b)/c) * ((a-b)/c) = ((a-b) * (a-b)) / (c * c) = (a * a - 2ab + b * b) / (c * c)
        const double barCPrimePow7 = barCPrime * barCPrime * barCPrime * barCPrime * barCPrime * barCPrime * barCPrime;
        const double R_C = 2.0 * sqrt(barCPrimePow7 / (barCPrimePow7 + pow25To7));

        const double S_L = 1.0 + ((0.015 * (barLPrime * barLPrime - (100 * barLPrime) + 2500.0)) /
            sqrt(20 + barLPrime * barLPrime - (100 * barLPrime) + 2500.0));
        const double S_C = 1.0 + (0.045 * barCPrime);
        const double S_H = 1.0 + (0.015 * barCPrime * T);

        const double R_T = (-sin(2.0 * deltaTheta)) * R_C;

        /*constexpr double k_L = 1.0, k_C = 1.0, k_H = 1.0;
        return sqrt(pow(deltaLPrime / (k_L * S_L), 2.0) +
                    pow(deltaCPrime / (k_C * S_C), 2.0) +
                    pow(deltaHPrime / (k_H * S_H), 2.0) +
                (R_T * (deltaCPrime / (k_C * S_C)) *
                       (deltaHPrime / (k_H * S_H))));*/

        result[i] = sqrt((deltaLPrime * deltaLPrime) / (S_L * S_L) +
            (deltaCPrime * deltaCPrime) / (S_C * S_C) +
            (deltaHPrime * deltaHPrime) / (S_H * S_H) +
            (R_T * (deltaCPrime / S_C) *
                (deltaHPrime / S_H)));
    }
}

void CIEDE2000NewWrapper(float *colour1, float *colour2, double *result, int t_size)
{
    CIEDE2000New << <1, 1 >> > (colour1, colour2, result, t_size);
}

void CIEDE2000OldWrapper(float *colour1, float *colour2, double *result, int t_size)
{
    CIEDE2000Old << <1, 1 >> > (colour1, colour2, result, t_size);
}