#include <algorithm>
#include <math.h>
#include <stdio.h>



/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 */

float quick_select(float arr[], int n) 
// calculates median, fast

{
    int low, high ;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
	      std::swap(arr[low], arr[high]) ;
            return arr[median] ;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    std::swap(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       std::swap(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     std::swap(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    std::swap(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

	std::swap(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    std::swap(arr[low], arr[hh]);

    /* Re-set active partition */
    if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
}


float rfimean (float arr[], int n, int nsigma, float *cleanmean, float *outliermean, int *numbad) {
  float median=quick_select(arr,n);
  int nhalf=n/2;
  float sigma=0;
    for (int i=0; i<nhalf; i++) {
      float x=arr[i]-median;
      sigma+=x*x;
    }
    sigma=sqrt(sigma/nhalf);
    
    // now calculate two means
    float meanok=0;
    float meanbad=0;
    int numok=0;
    *numbad=0;
    float div=median+ nsigma*sigma;
    for (int i=0; i<n; i++) {
      float x=arr[i];
      if (x<div) {
	numok+=1;
	meanok+=x;
	  } else {
	*numbad+=1;
	meanbad+=x;
      }
    }

    *cleanmean=meanok/numok;

    if (meanbad>0)
      *outliermean=meanbad/(*numbad);
    else
      *outliermean=0;
    return median;
}

