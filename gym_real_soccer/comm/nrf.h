#ifndef NRF_H
#define NRF_H

/**\mainpage NRF class made by the RoboCIn for the project IEEE - VSS
 * @author RoboCin
 *
 * This class has a group of methods to be used for NRF between the main PC and all the robots.
 *
 */



#include <string.h>		// string function definitions
#include <unistd.h>		// UNIX standard function definitions
#include <fcntl.h>		// File control definitions
#include <errno.h>		// Error number definitions
#include <termios.h>	// POSIX terminal control definitions
#include <iostream>
#include <stdio.h>
//#include "spdlog/spdlog.h"

class NRF{
public:
    // Definitions
    #define DEFAULT_DEVICE_NAME_NRF         "/dev/NRF_Master"
    #define DEFAULT_BAUDRATE_NRF            B115200

    // Definitions
    #define NRF_BUFFER_SIZE             3
    #define _MSG_BEGIN                  2
    #define _MSG_END                    1

    /*
     * Class constructor
     */
    NRF(std::string device = DEFAULT_DEVICE_NAME_NRF, unsigned int baudrate = DEFAULT_BAUDRATE_NRF);

    /*
     * Class destructor
     */
    ~NRF();

    /*
     * Setup NRF
     * @param device File to use as stream. Like /dev/ttyUSB0 (std::string).
     * @param baudrate Channel baudrate (unsigned int).
     * @return 0 if everything is fine. Other number otherwise.
     */
    char setup();

    int setMessage(int start, int id, int motorSpeed1, int motorSpeed2, int kickRet, int batteryRet, int stop);

    int send(int start, int id, int motorSpeed1, int motorSpeed2, int kickRet, int batteryRet, int stop);

    int send(int id, int motorSpeed1, int motorSpeed2, int kickRet = 0, int batteryRet = 0);

    int send(int id, std::pair<double, double> speed, int kickRet = 0, int batteryRet = 0);

    void recv();

    void setRawBuffer(char b1, char b2, char b3);

    int sendRawBuffer();
    /*
     * Set the flag that indicates whther that class shall use the hardware to send messages or not.
     * It was devided into two functions for better method naming.
     */
    void startSending();
    void stopSending();

    /*
     * Set which NRF net the hardware will use, it makes possible change nets while running.
     */
    void setNet(int);
private:

    /*
     * Packs the nRF24L01+ protocol bits into 3 bytes
     * @return message packed into a byte (unsigned char).
     */
    unsigned char packB1();

    unsigned char packB2();

    unsigned char packB3();

    /// Flag that indicates whether the XBee class shall send the messages or not
    bool _shallSend;

    /// File descriptor of the USB connection with the nRF radio
    int		nrf_fd;

    /// Data structure containing terminal information
    struct termios _tty;

    /// Data structure to packet the messages using Robocin nRF24L01+ protocol
    typedef struct nrf_packet{
        //first byte
        struct B1
        {
            unsigned char start            : 2,
            id                             : 2,
            batteryRet                     : 1,
            kickRet                        : 1,
            motorSpeed1_p1                 : 2;
            unsigned char                  : 0;
        }b1;

        //second byte
        struct B2
        {
            unsigned char motorSpeed1_p2   : 6,
            motorSpeed2_p1                 : 2;
            unsigned char                  : 0;
        }b2;

        //third byte
        struct B3
        {
            unsigned char motorSpeed2_p2   : 6,
            stop                           : 2;
            unsigned char                  : 0;
        }b3;
    }nrfPacket;

    /// nRF packet instance
    nrfPacket packet;

    unsigned char buffer_write[4];
    unsigned char buffer_read[64];
    unsigned int first_read;

    /// Byte that indicates which NRF net interface will use
    char _NRFNet;

    std::string _device;
    int _baudrate;
};

#endif // NRF_H
