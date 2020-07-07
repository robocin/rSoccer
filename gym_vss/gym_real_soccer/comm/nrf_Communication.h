/**\mainpage Communication class made by the RoboCIn for the project IEEE - VSS
 * @author RoboCin
 *
 * This class has a group of methods to be used for communication between the main PC and all the robots using the nRF24L01+ radios.
 *
 */
#ifndef NRF_COMMUNICATION_H
#define NRF_COMMUNICATION_H


#include <string.h>		// string function definitions
#include <unistd.h>		// UNIX standard function definitions
#include <fcntl.h>		// File control definitions
#include <errno.h>		// Error number definitions
#include <termios.h>	// POSIX terminal control definitions
#include <iostream>
#include <stdio.h>
#include <stdint.h>
//#include "spdlog/spdlog.h"



class nrf_Communication {
public:

    // Definitions
    #define DEFAULT_DEVICE_NAME         "/dev/NRF_USB"
    #define DEFAULT_BAUDRATE            B115200
    #define NRF_BUFFER_SIZE             11
    #define _MSG_BEGIN                  2
    #define _MSG_END                    1
    #define MSG_RETURN_POSITION         4
    #define MSG_RETURN_BATTERY          5
    #define LENGHT_SPEED                5
    #define LENGHT_PID                  11
    #define LENGHT_POSITIONS            11

    /*
     * Class constructor
     */
    nrf_Communication();

    /*
     * Class destructor
     */
    ~nrf_Communication();


    /**
     * Structure for the first message format,
     * this type sends the left and right motor speeds
     * is also sending the message type, the robot id
     * that should receive the message and has space
     * to send flags to the robot.
     */

    typedef struct typeSpeed{
        uint8_t typeMsg:4;
	    uint8_t id:4;
        int8_t leftSpeed:8;
        int8_t rightSpeed:8;
        uint8_t flags:8;
        uint8_t end:8;
	} __attribute__((packed))TypeSpeed;

	typedef union msgSpeed{
        unsigned char encoded[5];
        TypeSpeed decoded;
	} MsgSpeed;


    /**
    * Structure for the second message format,
    * this type sends the current x and y positions and the current angle,
    * the x and y target positions and the target angle
    * is also sending the message type, the robot ID
    * that should receive the message and have space
    * to send flags to the robot.
    */

	typedef struct typePositions{
        uint8_t typeMsg:4;
        uint8_t id:4;
        uint8_t curPosX:8;
        uint8_t curPosY:8;
        int16_t curAngle:16;
        uint8_t objPosX:8;
        uint8_t objPosY:8;
        int16_t objAngle:16;
        uint8_t flags:8;
        uint8_t end:8;
	}  __attribute__((packed)) TypePositions;

	typedef union msgPositions{
        unsigned char encoded[11];
        TypePositions decoded;
	} MsgPositions;


    /**
     * Structure for the third message format,
     * this type sends values of kp, ki, kd and alfaS
     * to set the PID
     * is also sending the message type, the robot ID
     * that should receive the message and have space
     * to send flags to the robot.
     */

	typedef struct typePID{
        uint8_t typeMsg:4;
        uint8_t id:4;
        uint16_t kp:16;
        uint16_t ki:16;
        uint16_t kd:16;
        uint16_t alfa:16;
        uint8_t flags:8;
        uint8_t end:8;
	} __attribute__((packed)) TypePID;

	typedef union msgPID{
	  unsigned char encoded[11];
	  TypePID decoded;
	} MsgPID;


    /**
     * Structure for the fourth message format,
     * this type is a return, in it comes values of the
     * current positions of the robot and its angle
     * the message type and the robot ID are also received
     *who sent the message
     */

	typedef struct typeRetPosition{
	  uint8_t typeMsg:4;
	  uint8_t id:4;
	  uint16_t curPosX:16;
	  uint16_t curPosY:16;
	  int16_t curAngle:16;
	}  __attribute__((packed)) TypeRetPosition;

	typedef union msgRetPosition{
	  unsigned char encoded[7];
	  TypeRetPosition decoded;
	} MsgRetPosition;


    /**
     * Structure for the fifth message format,
     * this type is return, in it comes the battery level of the robot
     * the message type and the robot ID are also received
     * who sent the message
     */

	typedef struct typeRetBattery{
		uint8_t typeMsg:4;
		uint8_t id:4;
		uint8_t batteryLevel:8;
	} __attribute__((packed)) TypeRetBattery;

	typedef union msgRetBattery{
	  unsigned char encoded[2];
	  TypeRetBattery decoded;
	} MsgRetBattery;


    /**
     * Structure to return the values that the robot sent
     * upon receiving request via the flag field
     */

    typedef struct returnedInfo{
        int id;
        bool position;
        int PosX;
        int PosY;
        float Angle;
        bool battery;
        float BatteryLevel;
    } ReturnedInfo;


    /**
     * Setup communication
     * @param device File to use as stream. Like /dev/ttyUSB0 (std::string).
     * @param baudrate Channel baudrate (unsigned int).
     * @return 0 if everything is fine. Other number otherwise.
     */
    int setup(std::string device = DEFAULT_DEVICE_NAME, unsigned int baudrate = DEFAULT_BAUDRATE);

    /**
     * Message configuration sends the speed
     * @param id
     * @param motorSpeed1
     * @param motorSpeed2
     * @param flags
     */
    void setSpeed(uint8_t id,int8_t motorSpeed1,int8_t motorSpeed2,uint8_t flags);

    /**
     * Message configuration sends the positions
     * @param id
     * @param cur_pos_x
     * @param cur_pos_y
     * @param cur_angle
     * @param obj_pos_x
     * @param obj_pos_y
     * @param obj_angle
     * @param flags
     */
    void setPosition(uint8_t id, uint8_t cur_pos_x, uint8_t cur_pos_y, int16_t cur_angle, uint8_t obj_pos_x, uint8_t obj_pos_y, int16_t obj_angle,uint8_t flags);

    /**
     * Message configuration sends the PID configuration
     * @param id
     * @param kp
     * @param ki
     * @param kd
     * @param alfa
     * @param flags
     */
    void setConfigurationPID(uint8_t id, double kp, double ki, double kd, double alfa, uint8_t flags);

    /**
     * Configuration of message of type Speed and sending of same
     * @param id
     * @param motorSpeed1
     * @param motorSpeed2
     * @param flags
     * @return number of bytes written to serial port
     */
    int sendSpeed(uint8_t id,int8_t motorSpeed1,int8_t motorSpeed2,uint8_t flags);

    /**
     * Configuration of message of type position and sending of same
     * @param id
     * @param cur_pos_x
     * @param cur_pos_y
     * @param cur_angle
     * @param obj_pos_x
     * @param obj_pos_y
     * @param obj_angle
     * @param flags
     * @return number of bytes written to serial port
     */
    int sendPosition(uint8_t id, uint8_t cur_pos_x, uint8_t cur_pos_y, float cur_angle, uint8_t obj_pos_x, uint8_t obj_pos_y, float obj_angle,uint8_t flags);

    /**
     * Configuration of message of type PID configuration and sending of same
     * @param id
     * @param kp
     * @param ki
     * @param kd
     * @param alfa
     * @param flags
     * @return number of bytes written to serial port
     */
    int sendConfigurationPID(uint8_t id, double kp, double ki, double kd, double alfa, uint8_t flags);

    /**
     * Read data from the serial port, verify the validity of the received data
     * and separate the data according to the type
     * @return message type read from serial port
     */
    int recv();

    /**
     * Returns the data received from the robots that were requested by
     * the computer in the form of a structure
     * @param id
     * @return
     */
    ReturnedInfo getInfoRet(int id);

private:
    /// File descriptor of the USB connection with the nRF radio
	int nrf_fd;

    /// Data structure containing terminal information
    struct termios _tty;

    /// Declaration of message types
    MsgSpeed _mSpeed;
    MsgPositions _mPositions;
    MsgPID _mPID;
    MsgRetPosition _rcvPosition;
    MsgRetBattery _rcvBattery;
    ReturnedInfo _retInf[3];

    int _returnRead;
    int _typeOfReturn;

    unsigned char buffer_write[12];
    unsigned char buffer_read[10];
    unsigned int first_read;
};

#endif
