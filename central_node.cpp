/*
     Description: A node for sending and receiving sensorimotor data from the NAO robot.
     Modified by Mehdi Tlili for the homework of the lecture:
     Biologically inspired learning for humanoid robotics (BILHR)
 */


#include <ros/ros.h>

// ROS and OpenCV image processing
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include "std_msgs/String.h"
#include <fstream>
// own files
#include "robot_config.h"

#include "cmac.h"
#include <time.h>
#include <math.h>


using namespace std;
using namespace cv;
using namespace sensor_msgs;
using namespace message_filters;

namespace enc = sensor_msgs::image_encodings;


// subscribers to tactile and touch sensors
ros::Subscriber tactile_sub;
ros::Subscriber bumper_sub;

// publisher to robot joints for sending the target joint angles
ros::Publisher target_joint_state_pub;

// joint stiffnesses
ros::Publisher stiffness_pub;

// received motor state of the HEAD
double motor_head_in[HEAD_JOINTS];

// received motor state of the LEFT ARM
double motor_l_arm_in[L_ARM_JOINTS];

// received motor state of the RIGHT ARM
double motor_r_arm_in[R_ARM_JOINTS];

// label of the GUI window showing the raw image of NAO's camera
static const char cam_window[] = "NAO Camera (raw image)";


//========================By Mehdi Tlili===========================//
//Flag for knowing which button was pushed last
int T2pushed = 0;
int T3pushed = 0;
int counterLArm = 0;
double initPosLArm[L_ARM_JOINTS];
double tempPos[L_ARM_JOINTS];
Point ballPos;
bool isSavingData;
bool isTestingData;
//=======================By Mehdi Tlili ==============================//





//==============================Added by Mehdi Tlili====================//
    //Parameters for the CMAC neural network
    int res = 50;
    int na = 5;
    int ny = 2;
    int nx = 2;
    double gamma_cmac = 0.1;
  //Create cmac class
    cmac nao(nx,ny,na,0,320,240,res);
//==============================Added by Mehdi Tlili====================//









//==============================BY Mehdi TLILI =====================//

// set the stiffness

void setStiffnessHead(float value)
{
    cout << "setting stiffnesses (head) to " << value << endl;

    BILHR_ros::JointState target_joint_stiffness;

    // set stiffnesses of HEAD joints
    target_joint_stiffness.name.clear();
    target_joint_stiffness.name.push_back("Head");
    target_joint_stiffness.effort.clear();
    for (int i=0; i<HEAD_JOINTS; i++)
        target_joint_stiffness.effort.push_back(value);

    stiffness_pub.publish(target_joint_stiffness);

}
//==============================Added by Mehdi Tlili====================//



//==============================Added by Mehdi Tlili====================//
// set the stiffness for Left Arm
void setStiffnessLArm(float value)
{
    cout << "setting stiffnesses (LArm) to " << value << endl;

    BILHR_ros::JointState target_joint_stiffness;

    // set stiffnesses of LArm joints
    target_joint_stiffness.name.clear();
    target_joint_stiffness.name.push_back("LArm");
    target_joint_stiffness.effort.clear();
    for (int i=0; i<L_ARM_JOINTS; i++)
        target_joint_stiffness.effort.push_back(value);

    stiffness_pub.publish(target_joint_stiffness);

}
//==============================Added by Mehdi Tlili====================//


//==============================Added by Mehdi Tlili====================//
void setStiffnessLArmCMAC(float value)
{
    cout << "setting stiffnesses (LArm) to " << value << endl;

    BILHR_ros::JointState target_joint_stiffness;

    // set stiffnesses of LArm joints
    target_joint_stiffness.name.clear();
    target_joint_stiffness.name.push_back("LArm");
    target_joint_stiffness.effort.clear();
    for (int i=0; i<2; i++)
        target_joint_stiffness.effort.push_back(0);
    for (int i=2; i<L_ARM_JOINTS; i++)
        target_joint_stiffness.effort.push_back(0.9);

    stiffness_pub.publish(target_joint_stiffness);

}
//==============================Added by Mehdi Tlili====================//



//==============================Added by Mehdi Tlili====================//
void setStiffnessRArm(float value)
{
    cout << "setting stiffnesses (RArm) to " << value << endl;

    BILHR_ros::JointState target_joint_stiffness;

    // set stiffnesses of LArm joints
    target_joint_stiffness.name.clear();
    target_joint_stiffness.name.push_back("RArm");
    target_joint_stiffness.effort.clear();
    for (int i=0; i<R_ARM_JOINTS; i++)
        target_joint_stiffness.effort.push_back(value);

    stiffness_pub.publish(target_joint_stiffness);

}
//==============================Added by Mehdi Tlili====================//




//==============================BY Mehdi TLILI =====================//
//Function to Save training data for CMAC
void saveTrainingData(const BILHR_ros::JointState::ConstPtr& joint_state)
{
    // buffer for incoming message
    std_msgs::Float32MultiArray buffer;


    // extract the proprioceptive state of the LEFT ARM
    buffer.data.clear();
    for (int i=0; i<ROBOT_JOINTS; i++)
    {
        if (joint_state->name[i] == "LShoulderPitch")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "LShoulderRoll")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }

    }


    //Save the training data containing ball position and arm position
    if(ballPos.x >0 && ballPos.y >0)
        {
            ofstream myfile;
            myfile.open("/home/pcnao03/trainingData.txt",ios_base::app);
            cout<<buffer.data.at(0)<<"\t"<<buffer.data.at(1)<<"\t"<<ballPos.x<<"\t"<<ballPos.y<<endl;
            myfile <<buffer.data.at(0)<<"\t"<<buffer.data.at(1)<<"\t"<<ballPos.x<<"\t"<<ballPos.y<<"\n";
            myfile.close();
        }

}
//==============================Added by Mehdi Tlili====================//




// callback function for vision
void visionCB(const sensor_msgs::ImageConstPtr& msg)
{
    // pointer on OpenCV image

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    Mat HSVImage;
    Mat ThreshImage;
    try
    {
        // transform ROS image into OpenCV image
        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
        //==============================Added by Mehdi Tlili====================//
        //Transform the colors into HSV
        cvtColor(cv_ptr->image,HSVImage,CV_BGR2HSV);
        //for blue ball
        //inRange(HSVImage,Scalar(80,90,70),Scalar(110,125,200),ThreshImage);
        //for red ball
        inRange(HSVImage,Scalar(170,160,60),Scalar(180,256,256),ThreshImage);

        SimpleBlobDetector *blobs;
        SimpleBlobDetector::Params params;
        params.minArea = 50.0f;
        params.filterByArea = true;
        params.filterByInertia = false;
        params.filterByColor = false;
        params.filterByConvexity = false;
        params.filterByCircularity = false;
        params.minDistBetweenBlobs = 50.0f;

        blobs = new SimpleBlobDetector(params);
        blobs->create("SimpleBlobDetector");
        vector<cv::KeyPoint> keypoints;
        blobs->detect(ThreshImage,keypoints);
        //printf("Size of blob = %d\n",keypoints.size());
        double c_x = 0;
        double c_y = 0;
        //get centroid of the blob
        if(keypoints.size() >0)
        {
            for(int i = 0;i<keypoints.size();i++)
            {
                c_x+=keypoints[i].pt.x;
                c_y+=keypoints[i].pt.y;
            }
            c_x/=keypoints.size();
            c_y/=keypoints.size();
            Point pt;
            pt.x = (int)c_x;
            pt.y = (int)c_y;
            //For training
            ballPos.x = (int)c_x;
            ballPos.y = (int)c_y;
            circle(ThreshImage,pt,20,Scalar(255,0,0),3);
        }
        //==============================Added by Mehdi Tlili====================//


    }
    catch (cv_bridge::Exception& e)		// throw an error msg. if conversion fails
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    //show the raw camera image
    //imshow(cam_window, cv_ptr->image);
    //show filtered HSV image containing the blob
    imshow(cam_window,ThreshImage);
    waitKey(1);
}




// callback function for bumpers
void bumperCB(const BILHR_ros::Bumper::ConstPtr& __bumper)
{
    // check each bumper
    cout << "bumper " << (int)__bumper->bumper << endl;
    static bool left_bumper_flag = false;
    static bool right_bumper_flag = false;
    // check left bumper
    if (((int)__bumper->bumper == 1) && ((int)__bumper->state == 1))
    {
        left_bumper_flag = !left_bumper_flag;   // toggle flag
        // do something, e.g.:
        // set / reset stiffness
        if (left_bumper_flag)
            setStiffnessHead(0.005);
            setStiffnessLArm(0.005);
            setStiffnessRArm(0.005);
            //T2pushed = 0;
            //T1pushed = 0;
//        else
//            setStiffness(0.9);
    }

    // check right bumper
    if (((int)__bumper->bumper == 0) && ((int)__bumper->state == 1))
    {
        //==============================Added by Mehdi Tlili====================//
        right_bumper_flag = !right_bumper_flag;     // toggle flag
        setStiffnessHead(0.8);
        //==============================Added by Mehdi Tlili====================//
    }

}

// send commanded joint positions of the HEAD
void sendTargetJointStateHead(double* coordinates)
{
    //double dummy[HEAD_JOINTS];  // dummy representing the comanded joint state
    BILHR_ros::JointAnglesWithSpeed target_joint_state;

    // specify the limb
    target_joint_state.joint_names.clear();
    target_joint_state.joint_names.push_back("Head");

    // specifiy the angle
    target_joint_state.joint_angles.clear();
    for (int i=0; i<HEAD_JOINTS; i++)
        target_joint_state.joint_angles.push_back(coordinates[i] /* array containing result */);

    // set speed
    target_joint_state.speed = 0.2;

    // set the mode of joint change
    target_joint_state.relative = 0;

    // send to robot
    target_joint_state_pub.publish(target_joint_state);

}




//==============================Added by Mehdi Tlili====================//
// send commanded joint positions of the LEFT ARM
void sendTargetJointStateLArm(double *coordinates)
{
    BILHR_ros::JointAnglesWithSpeed target_joint_state;

    // specify the limb
    target_joint_state.joint_names.clear();
    target_joint_state.joint_names.push_back("LArm");

    // specifiy the angle
    target_joint_state.joint_angles.clear();
    for (int i=0; i<L_ARM_JOINTS; i++)
        target_joint_state.joint_angles.push_back(coordinates[i] /* array containing result */);

    // set speed
    target_joint_state.speed = 0.2;

    // set the mode of joint change
    target_joint_state.relative = 0;

    // send to robot
    target_joint_state_pub.publish(target_joint_state);
    cout << "Sent new position for left arm" << endl;
}
//==============================Added by Mehdi Tlili====================//



//==============================Added by Mehdi Tlili====================//
void sendTargetJointStateRArm(double *coordinates)
{
    BILHR_ros::JointAnglesWithSpeed target_joint_state;

    // specify the limb
    target_joint_state.joint_names.clear();
    target_joint_state.joint_names.push_back("RArm");

    // specifiy the angle
    target_joint_state.joint_angles.clear();
    for (int i=0; i<R_ARM_JOINTS; i++)
        target_joint_state.joint_angles.push_back(coordinates[i] /* array containing result */);

    // set speed
    target_joint_state.speed = 0.2;

    // set the mode of joint change
    target_joint_state.relative = 0;

    // send to robot
    target_joint_state_pub.publish(target_joint_state);
    cout << "Sent new position for right arm" << endl;
}
//==============================Added by Mehdi Tlili====================//



// callback function for the head joints
void jointStateCB(const BILHR_ros::JointState::ConstPtr& joint_state)
{
    // buffer for incoming message
    std_msgs::Float32MultiArray buffer;

    // index
    int idx;


    // extract the proprioceptive state of the HEAD
    buffer.data.clear();
    for (int i=0; i<ROBOT_JOINTS; i++)
    {
        if (joint_state->name[i] == "HeadYaw")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << ": " << joint_state->position[i] << endl;
        }
        if (joint_state->name[i] == "HeadPitch")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << ": " << joint_state->position[i] << endl;
        }
    }

    // write data into array
    idx = 0;
    for(vector<float>::const_iterator iter = buffer.data.begin(); iter != buffer.data.end(); ++iter)
    {
        // store into temporary target motor state buffer
        motor_head_in[idx] = *iter;
        idx++;
    }

    // display data on terminal
    /*cout << "Head joints:  ";
    for (int i=0; i<HEAD_JOINTS; i++)
        cout << motor_head_in[i] << " ";
    cout << endl;*/


    // extract the proprioceptive state of the LEFT ARM
    buffer.data.clear();
    for (int i=0; i<ROBOT_JOINTS; i++)
    {
        if (joint_state->name[i] == "LShoulderPitch")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "LShoulderRoll")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "LElbowYaw")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "LElbowRoll")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "LWristYaw")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "LHand")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }

    }

    // write data into array
    idx = 0;
    for(vector<float>::const_iterator iter = buffer.data.begin(); iter != buffer.data.end(); ++iter)
    {
        // store into temporary target motor state buffer
        motor_l_arm_in[idx] = *iter;
        idx++;
    }


    // extract the proprioceptive state of the RIGHT ARM
    buffer.data.clear();
    for (int i=0; i<ROBOT_JOINTS; i++)
    {
        if (joint_state->name[i] == "RShoulderPitch")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "RShoulderRoll")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "RElbowYaw")
        {
           buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "RElbowRoll")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "RWristYaw")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }
        if (joint_state->name[i] == "RHand")
        {
            buffer.data.push_back(joint_state->position[i]);
            // cout << joint_state->name[i] << endl;
        }

    }

    // write data into array
    idx = 0;
    for(vector<float>::const_iterator iter = buffer.data.begin(); iter != buffer.data.end(); ++iter)
    {
        // store into temporary target motor state buffer
        motor_r_arm_in[idx] = *iter;
        idx++;
    }


    //==============================Added by Mehdi Tlili====================//
    //Save training data
     if(isSavingData)
     {
         printf("Saving Data\n");
         saveTrainingData(joint_state);
         isSavingData = false;
     }
     //Test trained CMAC with the ball
     if(isTestingData)
     {
         setStiffnessLArm(0.9);
         printf("Testing Data\n");
         double output1,output2;
         double coordinates[6] = {0,0,0,0,-3.14,0};
         if(ballPos.x >0 && ballPos.y >0)
         {
            nao.map(ballPos.x, ballPos.y,&output1,&output2);
            cout<<"Joint1 = "<<output1<<endl<<"Joint2 = "<<output2<<endl;
            coordinates[0] = output1;
            coordinates[1] = output2;
            sendTargetJointStateLArm(coordinates);
         }
     }
     //==============================Added by Mehdi Tlili====================//
}




//callback function for tactile buttons (TBs) on the head
void tactileCB(const BILHR_ros::TactileTouch::ConstPtr& __tactile_touch)
{
    // check TB 3 (rear)
    if (((int)__tactile_touch->button == 3) && ((int)__tactile_touch->state == 1))
    {
        cout << "TB " << (int)__tactile_touch->button << " touched" << endl;
        //==============================Added by Mehdi Tlili====================//
        double a, b,c,d;
        //Choose if for generating new training data or training CMAC with existing data
        bool forTraining = true;
        if(forTraining)
        {
            //Train CMAC From saved training data and do 100 iterations
            for(int iter = 0;iter<100;iter++)
            {
                //cout<<"Iter training = "<<iter<<endl;
                std::string line;
                std::ifstream myfile ;
                myfile.open("/home/pcnao03/trainingData.txt");
                double x;
                //cout<<"if file opened " <<myfile.is_open()<<endl;
                while (!myfile.eof())
                {
                    std::getline(myfile, line);
                    //cout<<line<<endl;
                    std::istringstream iss(line);
                    if (!(iss >> a >> b >>c >>d )) { break; }
                    cout <<"y1 = "<<c<<" y2 = "<<d<<endl;
                    try
                    {
                    nao.train(c,d,a,b);
                    }
                    catch(int e)
                    {
                        cout<<"error detected"<<endl;
                    }
                }
                myfile.close();
           }
           cout <<"Training finished"<<endl;
        }
        else
        {
            //Save new training data
            isSavingData = true;
        }
        //==============================Added by Mehdi Tlili====================//
    }

    // check TB 2 (middle)
    if (((int)__tactile_touch->button == 2) && ((int)__tactile_touch->state == 1))
    {
        cout << "TB " << (int)__tactile_touch->button << " touched" << endl;
        //==============================Added by Mehdi Tlili====================//
        //Put Nao in initial position for training or testing
        setStiffnessHead(0.9);
        setStiffnessLArmCMAC(0);
        double tmp[6] = {0, 0, 0, 0, -3.14,0};
        sendTargetJointStateLArm(tmp);
        double tmp2[2] = {3.14/4 ,0};
        sendTargetJointStateHead(tmp2);
        //==============================Added by Mehdi Tlili====================//

    }

    // check TB 1 (front)
    if (((int)__tactile_touch->button == 1) && ((int)__tactile_touch->state == 1))
    {
        cout << "TB " << (int)__tactile_touch->button << " touched" << endl;
        //==============================Added by Mehdi Tlili====================//
        //Testing only when pushing button
       /* setStiffnessLArm(0.9);
        printf("Testing Data\n");
        double output1,output2;
        double coordinates[6] = {0,0,0,0,-3.14,0};
        if(ballPos.x >0 && ballPos.y >0)
        {         
           nao.map(ballPos.x, ballPos.y,&output1,&output2);
           cout<<"Joint1 = "<<output1<<endl<<"Joint2 = "<<output2<<endl;
           coordinates[0] = output1;
           coordinates[1] = output2;
           sendTargetJointStateLArm(coordinates);
        }*/

        //continuous testing, makes the robot reach for the ball with his left arm continously
        isTestingData = !isTestingData;
        //==============================Added by Mehdi Tlili====================//
    }

}

// send commanded joint positions of the RIGHT ARM
void sendTargetJointStateRArm(/* maybe a result as function argument */)
{
    double dummy[R_ARM_JOINTS];  // dummy representing the comanded joint state
    BILHR_ros::JointAnglesWithSpeed target_joint_state;

    // specify the limb
    target_joint_state.joint_names.clear();
    target_joint_state.joint_names.push_back("RArm");

    // specifiy the angle
    target_joint_state.joint_angles.clear();
    for (int i=0; i<R_ARM_JOINTS; i++)
        target_joint_state.joint_angles.push_back(dummy[i] /* array containing result */);

    // set speed
    target_joint_state.speed = 0.2;

    // set the mode of joint change
    target_joint_state.relative = 0;

    // send to robot
    target_joint_state_pub.publish(target_joint_state);

}


// callback function for key events
void keyCB(const std_msgs::String::ConstPtr& msg)
{
    ROS_INFO("key pushed: %s", msg->data.c_str());

    // start the robot behaviour
    if (*(msg->data.c_str()) == '0')
	{
		cout << "keyCB()" << endl;  	
	}
}

int main(int argc, char** argv)
{
    //Init node
    ros::init(argc, argv, "central_node");
    ros::NodeHandle central_node_nh;

    // messaging with the NAO nodes

    // advertise joint stiffnesses
    stiffness_pub = central_node_nh.advertise<BILHR_ros::JointState>("joint_stiffness", 1);
    // subscribe to the joint states
    // the topic is the same as the one of the wrapper node of the NAO robot
    ros::Subscriber joint_state_sub;
    joint_state_sub = central_node_nh.subscribe("joint_states", 1, &jointStateCB);

    // advertise the target joint states
    target_joint_state_pub = central_node_nh.advertise<BILHR_ros::JointAnglesWithSpeed>("joint_angles", 1);    // to NAO robot

    // using image_transport to publish and subscribe to images
    image_transport::ImageTransport image_tran(central_node_nh);

    // subscribe to the raw camera image
    image_transport::Subscriber image_sub;
    image_sub = image_tran.subscribe("image_raw", 1, &visionCB);

    // subscribe to tactile and touch sensors
    tactile_sub = central_node_nh.subscribe("tactile_touch", 1, tactileCB);
    bumper_sub = central_node_nh.subscribe("bumper", 1, bumperCB);

    // set up the subscriber for the keyboard
    ros::Subscriber key_sub;
    key_sub = central_node_nh.subscribe("key", 5, keyCB);


    // create a GUI window for the raw camera image
    namedWindow(cam_window, 0);
    isTestingData = false;
    ros::spin();
    return 0;
}
