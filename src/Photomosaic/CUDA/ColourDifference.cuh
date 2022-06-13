#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

typedef double(*p_dfColourDifference)(float *, float *);

//Calculates the euclidean difference between two 3-channel colour values
__device__ inline double euclideanDifference(float *colour1, float *colour2)
{
    return sqrt(pow((double)(colour1[0] - colour2[0]), (double)2.0) +
        pow((double)(colour1[1] - colour2[1]), (double)2.0) +
        pow((double)(colour1[2] - colour2[2]), (double)2.0));
}

//Converts degrees to radians
__device__ inline constexpr double degToRadKernel(const double deg)
{
    return (deg * CUDART_PI) / 180;
}

//Calculates the CIEDE2000 difference between two 3-channel colour values
__device__ inline double CIEDE2000Difference(float *colour1, float *colour2)
{
    constexpr double deg360InRad = degToRadKernel(360.0);
    constexpr double deg180InRad = degToRadKernel(180.0);
    constexpr double pow25To7 = 6103515625.0; //pow(25, 7)

    const double C1 = sqrt((double)(colour1[1] * colour1[1]) + (colour1[2] * colour1[2]));
    const double C2 = sqrt((double)(colour2[1] * colour2[1]) + (colour2[2] * colour2[2]));
    const double barC = (C1 + C2) / 2.0;

    const double G = 0.5 * (1 - sqrt(pow(barC, (double)7.0) / (pow(barC, (double)7.0) + pow25To7)));

    const double a1Prime = (1.0 + G) * colour1[1];
    const double a2Prime = (1.0 + G) * colour2[1];

    const double CPrime1 = sqrt((a1Prime * a1Prime) + (colour1[2] * colour1[2]));
    const double CPrime2 = sqrt((a2Prime * a2Prime) + (colour2[2] * colour2[2]));

    double hPrime1;
    if (colour1[2] == 0 && a1Prime == 0.0)
        hPrime1 = 0.0;
    else
    {
        hPrime1 = atan2((double)colour1[2], a1Prime);
        //This must be converted to a hue angle in degrees between 0 and 360 by
        //addition of 2 pi to negative hue angles.
        if (hPrime1 < 0)
            hPrime1 += deg360InRad;
    }

    double hPrime2;
    if (colour2[2] == 0 && a2Prime == 0.0)
        hPrime2 = 0.0;
    else
    {
        hPrime2 = atan2((double)colour2[2], a2Prime);
        //This must be converted to a hue angle in degrees between 0 and 360 by
        //addition of 2pi to negative hue angles.
        if (hPrime2 < 0)
            hPrime2 += deg360InRad;
    }

    const double deltaLPrime = colour2[0] - colour1[0];
    const double deltaCPrime = CPrime2 - CPrime1;

    double deltahPrime;
    const double CPrimeProduct = CPrime1 * CPrime2;
    if (CPrimeProduct == 0.0)
        deltahPrime = 0;
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

    const double barLPrime = (colour1[0] + colour2[0]) / 2.0;
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
    const double S_C = 1 + (0.045 * barCPrime);
    const double S_H = 1 + (0.015 * barCPrime * T);

    const double R_T = (-sin(2.0 * deltaTheta)) * R_C;

    //constexpr double k_L = 1.0, k_C = 1.0, k_H = 1.0;
    return (double)sqrt(pow(deltaLPrime / (/*k_L * */S_L), (double)2.0) +
        pow(deltaCPrime / (/*k_C * */S_C), (double)2.0) +
        pow(deltaHPrime / (/*k_H * */S_H), (double)2.0) +
        (R_T * (deltaCPrime / (/*k_C * */S_C)) *
            (deltaHPrime / (/*k_H * */S_H))));
}