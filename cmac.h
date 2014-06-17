/*
   Implementation of the CMAC (Cerebellar Model Articulation Controller) neural network
   Author: Mehdi Tlili
*/

#ifndef CMAC_H
#define CMAC_H
#include <time.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

class cmac
{

public:
    cmac(int nx, int ny, int na, double minimum, double maximum1, double maximum2, int res);
    double hash(int p1,int p2);
    void quantizeAndAssociate(double y1,double y2);
    void map(double y1,double y2,double *out1,double *out2);
    void train(double y1,double y2, double x1, double x2);
private:
    //These variables have the same names as in the paper
    int _nx;
    int _ny;
    int _na;
    double _min;
    double _max1;
    double _max2;
    int _res;
    double *W1;
    double *W2;
    double d1[5];
    double d2[5];
    int mu_index[5];
};
#endif // CMAC_H



cmac::cmac(int nx,int ny,int na,double minimum,double maximum1,double maximum2,int res)
{
    //Nothing
    _nx = nx;
    _ny = ny;
    _na = na;
    _min = minimum;
    _max1 = maximum1;
    _max2 = maximum2;
    _res = res;
    int r = floor((res-2)/na)+2;
    W1 = new double[10000];
    W2 = new double[10000];
    srand (time(NULL));

    //Init weight vectors with random values between 0 and 1
    double tmp1,tmp2;
    for(int i=0;i<r+50;i++)
    {
        tmp1 = (double)rand()/RAND_MAX;
        tmp2 = (double)rand()/RAND_MAX;
        W1[i] = tmp1;
        W2[i] = tmp2;
    }

    //Init displacemenets (see paper)
    d1[0] = 3;
    d1[1] = 1;
    d1[2] = 4;
    d1[3] = 2;
    d1[4] = 0;
    d2[0] = 2;
    d2[1] = 3;
    d2[2] = 4;
    d2[3] = 0;
    d2[4] = 0;
}

//Mapping input ball position to output arm joins position
void cmac::map(double y1,double y2,double *out1,double *out2)
{

    quantizeAndAssociate(y1,y2);
    *out1  = 0;
    *out2 = 0;
    for(int i =0;i<_na;i++)
    {
        *out1+= W1[mu_index[i]];
        *out2+= W2[mu_index[i]];

    }
}
//Quantize input ball position to discrete values for input to neural network
void cmac::quantizeAndAssociate(double y1,double y2)
{
    //Check min max
    if(y1<_min)
        y1 = _min;
    if(y2 <_min)
        y2 = _min;
    if(y1 >_max1)
        y1 = _max1;
    if(y2 > _max2)
        y2 = _max2;
    double q1,q2;
    q1 = _res*(y1-_min)/(_max1-_min);
    q2 = _res*(y2-_min)/(_max2-_min);
    if(q1 >=_res)
        q1 = _res-1;
    if(q2>= _res)
        q2 = _res-1;

    int p1[_na];
    int p2[_na];
    for(int i = 0;i<_na;i++)
    //suppose we know we have 2 inputs
    {
        p1[i] = (q1+d1[i])/_na;
        p2[i] = (q2+d2[i])/_na;
        mu_index[i] = hash(p1[i],p2[i]);
    }

}

//For speed reasons, hashing is used in the internal implementation of the neural network
double cmac::hash(int p1,int p2)
{
    double r= floor((_res-2)/_na)+2;
    double h = 0;
        h+= h*r +p1;
        h+= h*r +p2;
    return h;
}
//Train the neural network using input ball position and output arm position
void cmac::train(double y1,double y2,double x1,double x2)
{
    double out1,out2,increment1,increment2;
    map(y1,y2,&out1,&out2);
    cout<<"output1 = "<<out1<<endl;
    cout<<"output2 = "<<out2<<endl;
    if(out1 <1000 && out2<1000)
    {

        increment1 = 0.1*(x1-out1)/_na;
        increment2 = 0.1*(x2-out2)/_na;
        for(int i = 0;i<_na;i++)
        {
            //cout<<"muindex = "<<mu_index[i];
            W1[mu_index[i]] += increment1;
            W2[mu_index[i]] += increment2;
        }
    }
}


