
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) finger_map.cpp -o finger_map$(python3-config --extension-suffix) $(pkg-config --cflags --libs yaml-cpp)
using namespace std;
#include <iostream>
#include <string>
#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "yaml-cpp/yaml.h"

#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
namespace py = pybind11;

// angle between 2 vectors in degrees
double angleBetweenVectors(double arr[], double brr[])
{
    double dot = arr[0] * brr[0] + arr[1] * brr[1] + arr[2] * brr[2];
    double lenSq1 = pow(arr[0], 2) + pow(arr[1], 2) + pow(arr[2], 2);
    double lenSq2 = pow(brr[0], 2) + pow(brr[1], 2) + pow(brr[2], 2);
    return acos(dot / sqrt(lenSq1 * lenSq2)) * 180 / M_PI;
}

// vector from 2 points
void vectorFromPoints(double pointP[], double pointQ[], double output[])
{
    for (int i = 0; i < 3; i++)
    {
        output[i] = pointQ[i] - pointP[i];
    }
}

// rotates point by theta degrees around x axis, y axis, z axis
void rotatebyDegrees(double point[], double output[], double theta, string input)
{
    // theta in radians
    theta = theta * M_PI / 180;
    if (input == "x_axis")
    {
        output[0] = point[0];
        output[1] = point[1] * cos(theta) + point[2] * -sin(theta);
        output[2] = point[1] * sin(theta) + point[2] * cos(theta);
    }
    else if (input == "y_axis")
    {
        output[0] = point[0] * cos(theta) + point[2] * sin(theta);
        output[1] = point[1];
        output[2] = point[0] * -sin(theta) + point[2] * cos(theta);
    }
    else
    {
        output[0] = point[0] * cos(theta) - point[1] * sin(theta);
        output[1] = point[0] * sin(theta) + point[1] * cos(theta);
        output[2] = point[2];
    }
}

void calculateMiddleAngles(double palm_in[], double mf_ee_in[], double mf_j1_in[], double mf_j2_in[], double mf_j3_in[], double angleTotal[], int wfing)
{
    double palm[3] = {}, mf_ee[3] = {}, mf_j1[3] = {}, mf_j2[3] = {}, mf_j3[3] = {};
    for (int i = 0; i < 3; i++)
    {
        palm[i] = palm_in[i];
        mf_ee[i] = mf_ee_in[i];
        mf_j1[i] = mf_j1_in[i];
        mf_j2[i] = mf_j2_in[i];
        mf_j3[i] = mf_j3_in[i];
    }

    for (int i = 0; i < 3; i++)
    { // normalizing palm to origin
        mf_ee[i] = mf_ee[i] - palm[i];
        mf_j1[i] = mf_j1[i] - palm[i];
        mf_j2[i] = mf_j2[i] - palm[i];
        mf_j3[i] = mf_j3[i] - palm[i];
        palm[i] = palm[i] - palm[i];
    }

    // angle betwen j3 and axis plane
    double plane_point[3] = {mf_j3[0], 0, 0};
    double plane_point2[3] = {mf_j3[0], mf_j3[1], 0};
    double curr_angle = angleBetweenVectors(plane_point2, plane_point);

    // retargeting to plane based on first angle
    double ee_r[3] = {}, j1_r[3] = {}, j2_r[3] = {}, j3_r[3] = {};
    rotatebyDegrees(mf_j3, j3_r, curr_angle, "z_axis");
    if (j3_r[1] > .00001 || j3_r[1] < -.00001)
    {
        curr_angle = 180 - curr_angle;
        rotatebyDegrees(mf_j3, j3_r, curr_angle, "z_axis");
    }
    rotatebyDegrees(mf_ee, ee_r, curr_angle, "z_axis");
    rotatebyDegrees(mf_j1, j1_r, curr_angle, "z_axis");
    rotatebyDegrees(mf_j2, j2_r, curr_angle, "z_axis");

    // angle between z plane and j3 to align j3 & palm vector to axis
    double plane_point3[3] = {j3_r[0], 0, 0};
    double plane_point4[3] = {j3_r[0], 0, j3_r[2]};
    double curr_angle2 = angleBetweenVectors(plane_point4, plane_point3);

    double ee_r2[3] = {}, j1_r2[3] = {}, j2_r2[3] = {}, j3_r2[3] = {};
    rotatebyDegrees(j3_r, j3_r2, curr_angle2, "y_axis");
    if (j3_r2[2] > .00001 || j3_r2[2] < -.00001)
    {
        curr_angle2 = 180 - curr_angle2;
        rotatebyDegrees(j3_r, j3_r2, curr_angle2, "y_axis");
    }
    rotatebyDegrees(ee_r, ee_r2, curr_angle2, "y_axis");
    rotatebyDegrees(j1_r, j1_r2, curr_angle2, "y_axis");
    rotatebyDegrees(j2_r, j2_r2, curr_angle2, "y_axis");

    // angle between y plane and j2
    double plane_point5[3] = {0, 0, j2_r2[2]};
    double plane_point6[3] = {0, j2_r2[1], j2_r2[2]};
    double curr_angle3 = angleBetweenVectors(plane_point5, plane_point6);

    // retargeting to plane based on third angle
    double ee_r3[3] = {}, j1_r3[3] = {}, j2_r3[3] = {};
    rotatebyDegrees(j2_r2, j2_r3, curr_angle3, "x_axis");
    if (j2_r3[1] > .00001 || j2_r3[1] < -.00001)
    {
        curr_angle3 = 180 - curr_angle3;
        rotatebyDegrees(j2_r2, j2_r3, curr_angle3, "x_axis");
    }
    rotatebyDegrees(ee_r2, ee_r3, curr_angle3, "x_axis");
    rotatebyDegrees(j1_r2, j1_r3, curr_angle3, "x_axis");

    // retarget to positive z values if below z
    if (ee_r3[2] < 0)
    {
        ee_r3[1] = -1 * ee_r3[1];
        ee_r3[2] = -1 * ee_r3[2];
        j1_r3[1] = -1 * j1_r3[1];
        j1_r3[2] = -1 * j1_r3[2];
        j2_r3[1] = -1 * j2_r3[1];
        j2_r3[2] = -1 * j2_r3[2];
    }

    // calculating finger vectors
    double ee_j1[3] = {}, j1_j2[3] = {}, j2_j3[3] = {}, j3_palm[3] = {};
    double theta_1, theta_2, theta_3, theta_4;
    vectorFromPoints(ee_r3, j1_r3, ee_j1);
    vectorFromPoints(j2_r3, j1_r3, j1_j2);
    vectorFromPoints(j2_r3, j3_r2, j2_j3);
    vectorFromPoints(palm, j3_r2, j3_palm);

    // finding first 2 angles
    theta_1 = angleBetweenVectors(ee_j1, j1_j2);
    theta_2 = angleBetweenVectors(j1_j2, j2_j3);

    // finding horizontal rotation
    double temp2[3] = {0, 0, j1_j2[2]};
    double temp3[3] = {0, j1_j2[1], j1_j2[2]};
    theta_4 = angleBetweenVectors(temp2, temp3);

    // adjust to left or right based on position on side of plane
    if (j1_r3[1] > 0)
    {
        theta_4 = 180 + theta_4;
    }
    else
    {
        theta_4 = 180 - theta_4;
    }

    // vertical rotation
    //  double temp4[3] = {j2_j3[0], 0, j2_j3[2]}; //j2 to j3 projected to y plane
    //  double temp5[3] = {j3_palm[0], 0, j3_palm[2]}; //palm to j3 projected to y plane
    theta_3 = angleBetweenVectors(j2_j3, j3_palm);

    // k value so fingers dont cross
    theta_4 = 180 + ((theta_4 - 180) / 10);
    if (wfing == 1)
    {
        angleTotal[0] = theta_1;
        angleTotal[1] = theta_2;
        angleTotal[2] = theta_3;
        angleTotal[3] = theta_4;
    }
    else if (wfing == 2)
    {
        angleTotal[5] = theta_1;
        angleTotal[6] = theta_2;
        angleTotal[7] = theta_3;
        angleTotal[8] = theta_4;
    }
    else if (wfing == 3)
    {
        angleTotal[9] = theta_1;
        angleTotal[10] = theta_2;
        angleTotal[11] = theta_3;
        angleTotal[12] = theta_4;
    }
    else if (wfing == 4)
    {
        angleTotal[13] = theta_1;
        angleTotal[14] = theta_2;
        angleTotal[15] = theta_3;
        angleTotal[16] = theta_4;
    }
    else if (wfing == 5)
    {
        angleTotal[17] = theta_1;
        angleTotal[18] = theta_2;
        angleTotal[19] = theta_3;
        angleTotal[20] = theta_4;
    }
    angleTotal[21] = 180;
    angleTotal[22] = 180;
}

// 63 and 16
py::array_t<double> retarget(py::array_t<double> input1)
{
    double insertHand[63];
    py::buffer_info buf1 = input1.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);

    for (long idx = 0; idx < buf1.shape[0]; idx++)
    {
        insertHand[idx] = ptr1[idx];
    }
    double *b = new double[16];

    double angletotal2[23] = {0};
    // double insertHand[63] = {0.7165310382843018,0.5979897379875183,9.177697961604281e-07,0.7422736883163452,0.5436157584190369,-0.044752318412065506,0.7543326020240784,0.487891286611557,-0.0768086165189743,0.7886183857917786,0.45565834641456604,-0.10837665945291519,0.8233191967010498,0.4371103048324585,-0.1408051997423172,0.6559509038925171,0.408966064453125,-0.05339256301522255,0.6180155873298645,0.33005282282829285,-0.08918295800685883,0.5916488766670227,0.2830435037612915,-0.11348428577184677,0.566082239151001,0.24210862815380096,-0.1290198117494583,0.618839681148529,0.45195701718330383,-0.05433397740125656,0.551618218421936,0.3999546468257904,-0.09343953430652618,0.5054413080215454,0.3734526038169861,-0.1166536882519722,0.4668273329734802,0.3528480529785156,-0.12744450569152832,0.6054028868675232,0.5079873204231262,-0.05872764810919762,0.5599067211151123,0.4928794801235199,-0.10809823125600815,0.6045299768447876,0.5200718641281128,-0.11564552038908005,0.6425783038139343,0.5382210612297058,-0.10597866773605347,0.60940021276474,0.5668247938156128,-0.06639498472213745,0.5807815790176392,0.561757504940033,-0.10494503378868103,0.610853374004364,0.5767804384231567,-0.10400904715061188,0.6398797035217285,0.585105299949646,-0.09274785220623016};
    double palm[] = {-24.158199310302734, 25.54400062561035, 1124.530029296875};

    double ThumbEE[] = {49.589500427246094, -19.278400421142578, 1015.4400024414062};
    double ThumbJ1[] = {33.4468994140625, -15.85949993133545, 1040.9200439453125};
    double ThumbJ2[] = {17.443500518798828, -14.036800384521484, 1070.4200439453125};
    double ThumbJ3[] = {2.7766199111938477, 7.481070041656494, 1097.0};

    double indexFingerEE[] = {32.980499267578125, 11.01990032196045, 960.2310180664062};
    double indexFingerJ1[] = {14.813400268554688, 5.958380222320557, 975.2369995117188};
    double indexFingerJ2[] = {-3.5343499183654785, 1.3145400285720825, 993.552001953125};
    double indexFingerJ3[] = {-15.350500106811523, -4.859789848327637, 1036.0699462890625};

    double middleFingerEE[] = {40.94540023803711, 39.09619903564453, 957.6060180664062};
    double middleFingerJ1[] = {21.188400268554688, 36.6427001953125, 972.2830200195312};
    double middleFingerJ2[] = {-2.6830599308013916, 32.85139846801758, 992.5250244140625};
    double middleFingerJ3[] = {-20.200000762939453, 22.54509925842285, 1035.989990234375};

    double ringFingerEE[] = {44.65489959716797, 57.91579818725586, 972.3980102539062};
    double ringFingerJ1[] = {25.188499450683594, 56.06420135498047, 986.9619750976562};
    double ringFingerJ2[] = {2.6755499839782715, 53.17250061035156, 1005.02001953125};
    double ringFingerJ3[] = {-14.769700050354004, 42.96839904785156, 1043.2900390625};

    double PinkyFingerEE[] = {30.85409927368164, 74.29889678955078, 997.0869750976562};
    double PinkyFingerJ1[] = {15.112500190734863, 72.39230346679688, 1009.75};
    double PinkyFingerJ2[] = {1.7122600078582764, 69.06610107421875, 1024.030029296875};
    double PinkyFingerJ3[] = {-6.992929935455322, 60.29119873046875, 1054.02001953125};

    palm[0] = insertHand[0];
    palm[1] = insertHand[1];
    palm[2] = insertHand[2];
    ThumbEE[0] = insertHand[12];
    ThumbEE[1] = insertHand[13];
    ThumbEE[2] = insertHand[14];
    ThumbJ1[0] = insertHand[9];
    ThumbJ1[1] = insertHand[10];
    ThumbJ1[2] = insertHand[11];
    ThumbJ2[0] = insertHand[6];
    ThumbJ2[1] = insertHand[7];
    ThumbJ2[2] = insertHand[8];
    ThumbJ3[0] = insertHand[3];
    ThumbJ3[1] = insertHand[4];
    ThumbJ3[2] = insertHand[5];

    middleFingerEE[0] = insertHand[36];
    middleFingerEE[1] = insertHand[37];
    middleFingerEE[2] = insertHand[38];
    middleFingerJ1[0] = insertHand[33];
    middleFingerJ1[1] = insertHand[34];
    middleFingerJ1[2] = insertHand[35];
    middleFingerJ2[0] = insertHand[30];
    middleFingerJ2[1] = insertHand[31];
    middleFingerJ2[2] = insertHand[32];
    middleFingerJ3[0] = insertHand[27];
    middleFingerJ3[1] = insertHand[28];
    middleFingerJ3[2] = insertHand[29];

    indexFingerEE[0] = insertHand[24];
    indexFingerEE[1] = insertHand[25];
    indexFingerEE[2] = insertHand[26];
    indexFingerJ1[0] = insertHand[21];
    indexFingerJ1[1] = insertHand[22];
    indexFingerJ1[2] = insertHand[23];
    indexFingerJ2[0] = insertHand[18];
    indexFingerJ2[1] = insertHand[19];
    indexFingerJ2[2] = insertHand[20];
    indexFingerJ3[0] = insertHand[15];
    indexFingerJ3[1] = insertHand[16];
    indexFingerJ3[2] = insertHand[17];

    ringFingerEE[0] = insertHand[48];
    ringFingerEE[1] = insertHand[49];
    ringFingerEE[2] = insertHand[50];
    ringFingerJ1[0] = insertHand[45];
    ringFingerJ1[1] = insertHand[46];
    ringFingerJ1[2] = insertHand[47];
    ringFingerJ2[0] = insertHand[42];
    ringFingerJ2[1] = insertHand[43];
    ringFingerJ2[2] = insertHand[44];
    ringFingerJ3[0] = insertHand[39];
    ringFingerJ3[1] = insertHand[40];
    ringFingerJ3[2] = insertHand[41];

    PinkyFingerEE[0] = insertHand[60];
    PinkyFingerEE[1] = insertHand[61];
    PinkyFingerEE[2] = insertHand[62];
    PinkyFingerJ1[0] = insertHand[57];
    PinkyFingerJ1[1] = insertHand[58];
    PinkyFingerJ1[2] = insertHand[59];
    PinkyFingerJ2[0] = insertHand[54];
    PinkyFingerJ2[1] = insertHand[55];
    PinkyFingerJ2[2] = insertHand[56];
    PinkyFingerJ3[0] = insertHand[51];
    PinkyFingerJ3[1] = insertHand[52];
    PinkyFingerJ3[2] = insertHand[53];

    // // tAngle(palm, ThumbEE, ThumbJ1, ThumbJ2, ThumbJ3);
    calculateMiddleAngles(palm, ThumbEE, ThumbJ1, ThumbJ2, ThumbJ3, angletotal2, 1);
    calculateMiddleAngles(palm, indexFingerEE, indexFingerJ1, indexFingerJ2, indexFingerJ3, angletotal2, 2);
    calculateMiddleAngles(palm, middleFingerEE, middleFingerJ1, middleFingerJ2, middleFingerJ3, angletotal2, 3);
    calculateMiddleAngles(palm, ringFingerEE, ringFingerJ1, ringFingerJ2, ringFingerJ3, angletotal2, 4);
    calculateMiddleAngles(palm, PinkyFingerEE, PinkyFingerJ1, PinkyFingerJ2, PinkyFingerJ3, angletotal2, 5);

    b[0] = angletotal2[8];
    b[1] = angletotal2[7];
    b[2] = angletotal2[6];
    b[3] = angletotal2[5];
    b[4] = angletotal2[12];
    b[5] = angletotal2[11];
    b[6] = angletotal2[10];
    b[7] = angletotal2[9];
    b[8] = angletotal2[16];
    b[9] = angletotal2[15];
    b[10] = angletotal2[14];
    b[11] = angletotal2[13];
    b[12] = angletotal2[3];
    b[13] = angletotal2[2];
    b[14] = angletotal2[1];
    b[15] = angletotal2[0];

    YAML::Node root = YAML::LoadFile("./cfg/teleop/teleop_config.yaml");
    for (int i = 0; i < 16; i++)
    {
        if (i == 0 || i == 4 || i == 8 || i == 12)
        {
            b[i] = b[i] - 180;
        }
        else
        {
            b[i] = (180 - b[i]);
        }
    }

    for (int i = 0; i < 16; i++)
    {
        b[i] = (b[i] * root["scale"][i].as<double>()) + (b[i] * root["offset"][i].as<double>());
        if (i == 0 || i == 4 || i == 8 || i == 12)
        {
            if (b[i] > 10)
            {
                b[i] = 10;
            }
            else if (b[i] < -10)
            {
                b[i] = -10;
            }
            b[i] = b[i] + 180; // undoing first loop
        }
        else
        {
            if (b[i] > 90)
            {
                b[i] = 90;
            }
            b[i] = -b[i] + 180;
        }
    }

    for (int i = 0; i < 16; i++)
    {
        if (i == 0 || i == 4 || i == 8 || i == 12)
        {
            b[i] = (abs((b[i] - 170) / 20) * ((root["max_bounds"][i].as<double>()) - root["min_bounds"][i].as<double>())) + root["min_bounds"][i].as<double>();
        }
        else
        {
            b[i] = (((180 - b[i]) / 90) * ((root["max_bounds"][i].as<double>()) - root["min_bounds"][i].as<double>())) + root["min_bounds"][i].as<double>();
        }
    }
    py::capsule free_when_done(b, [](void *f)
                               {
            double *b = reinterpret_cast<double *>(f);
            delete[] b; });

    return py::array_t<double>({16}, {8}, b, free_when_done);
}

PYBIND11_MODULE(finger_map, m)
{
    m.doc() = "pybind11 retarget function"; // optional module docstring
    m.def("retarget", &retarget, "Main Function");
}