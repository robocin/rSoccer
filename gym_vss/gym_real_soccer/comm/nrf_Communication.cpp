#include "nrf_Communication.h"

nrf_Communication::nrf_Communication(){
    //std::cout<< "nrf constructor" << std::endl;
}

nrf_Communication::~nrf_Communication(){
    //std::cout<< "nrf destructor" << std::endl;
}

int nrf_Communication::setup(std::string device, unsigned int baudrate){
    //std::cout << "setup nrf" << std::endl;

    // Open an USB descriptor
    // O_RDWR: Open file for read and write
    // O_NOCTTY: If set and path identifies a terminal device, open() shall not cause the terminal device to become the controlling terminal for the process
    // O_NONBLOCK: If O_NONBLOCK is set, the open() function shall return without blocking for the device to be ready or available. Subsequent behavior of the device is device-specific.
    this->nrf_fd = open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    

    // Clear terminal information structure
    memset (&this->_tty, 0, sizeof this->_tty);

    // Error Handling
    if (tcgetattr (this->nrf_fd, &this->_tty) != 0){
        //std::cout << "erro 1";
        return -1;
    }

    // Set Baud Rate
    cfsetospeed (&this->_tty, (speed_t)baudrate);
    cfsetispeed (&this->_tty, (speed_t)baudrate);


    // Setting other Port Stuff
    this->_tty.c_cflag     &=  ~PARENB;            // Make 8n1
    this->_tty.c_cflag     &=  ~CSTOPB;
    this->_tty.c_cflag     &=  ~CSIZE;
    this->_tty.c_cflag     |=  CS8;

    this->_tty.c_cflag     &=  ~CRTSCTS;           // no flow control
    this->_tty.c_cc[VMIN]   =  1;                  // read doesn't block
    this->_tty.c_cc[VTIME]  =  8;                  // 0.5 seconds read timeout
    this->_tty.c_cflag     |=  CREAD | CLOCAL;     // turn on READ & ignore ctrl lines

    // Make raw
    cfmakeraw(&this->_tty);

    // Flush Port, then applies attributes
    //tcsetattr(this->_USBDescriptor, TCSAFLUSH, &options);
    tcflush(this->nrf_fd, TCIFLUSH);
    if (tcsetattr (this->nrf_fd, TCSANOW, &this->_tty) != 0){
        //std::cout << "erro 2";
        return -2;
    }
    this->first_read = 1;
    return 0;
}

void nrf_Communication::setSpeed(uint8_t id,int8_t motorSpeed1,int8_t motorSpeed2,uint8_t flags){
    memset(this->_mSpeed.encoded,0,LENGHT_SPEED);

    this->_mSpeed.decoded.typeMsg = 0;
    this->_mSpeed.decoded.id = id;
    this->_mSpeed.decoded.leftSpeed = motorSpeed1;
    this->_mSpeed.decoded.rightSpeed = motorSpeed2;
    this->_mSpeed.decoded.flags = flags;
    this->_mSpeed.decoded.end = '!';
}

void nrf_Communication::setPosition(uint8_t id, uint8_t cur_pos_x, uint8_t cur_pos_y, int16_t cur_angle, uint8_t obj_pos_x, uint8_t obj_pos_y, int16_t obj_angle,uint8_t flags){
    memset(this->_mPositions.encoded,0,LENGHT_POSITIONS);

    this->_mPositions.decoded.typeMsg =  2;
    this->_mPositions.decoded.id =  id;
    this->_mPositions.decoded.curPosX =  cur_pos_x;
    this->_mPositions.decoded.curPosY = cur_pos_y;
    this->_mPositions.decoded.curAngle = cur_angle;
    this->_mPositions.decoded.objPosX = obj_pos_x;
    this->_mPositions.decoded.objPosY = obj_pos_y;
    this->_mPositions.decoded.objAngle = obj_angle;
    this->_mPositions.decoded.flags = flags;
    this->_mPositions.decoded.end = '!';
}

void nrf_Communication::setConfigurationPID(uint8_t id, double kp, double ki, double kd, double alfa, uint8_t flags){
    memset(this->_mPID.encoded,0,LENGHT_PID);

    this->_mPID.decoded.typeMsg = 3;
    this->_mPID.decoded.id = id;
    this->_mPID.decoded.kp = (uint16_t) kp*100;
    this->_mPID.decoded.ki = (uint16_t) ki*100;
    this->_mPID.decoded.kd = (uint16_t) kd*100;
    this->_mPID.decoded.alfa = (uint16_t) alfa*100;
    this->_mPID.decoded.flags = flags;
    this->_mPID.decoded.end = '!';
}

int nrf_Communication::sendSpeed(uint8_t id,int8_t motorSpeed1,int8_t motorSpeed2,uint8_t flags){
    this->setSpeed(id, motorSpeed1, motorSpeed2, flags);
    return write(this->nrf_fd, this->_mSpeed.encoded, sizeof this->_mSpeed.encoded);
}

int nrf_Communication::sendPosition(uint8_t id, uint8_t cur_pos_x, uint8_t cur_pos_y, float cur_angle, uint8_t obj_pos_x, uint8_t obj_pos_y, float obj_angle,uint8_t flags){
    int16_t angleOne;
    int16_t angleTwo;

    angleOne = (int16_t) (cur_angle*10);
    angleTwo = (int16_t) (obj_angle*10);

    this->setPosition(id, cur_pos_x, cur_pos_y, angleOne, obj_pos_x, obj_pos_y, angleTwo, flags);

    return write(this->nrf_fd, this->_mPositions.encoded, sizeof this->_mPositions.encoded);
}

int nrf_Communication::sendConfigurationPID(uint8_t id, double kp, double ki, double kd, double alfa, uint8_t flags){
    this->setConfigurationPID( id, kp, ki, kd, alfa,flags);
    return write(this->nrf_fd, this->_mPID.encoded, sizeof this->_mPID.encoded);
}

int nrf_Communication::recv(){
    memset(this->buffer_read, 0, 10);

    this->_returnRead = read(this->nrf_fd, this->buffer_read, sizeof this->buffer_read);
    //printf("Chegou: %d",this->_returnRead);
    for(int i = 0; i<10;i++ ){
        //printf("%d\n",buffer_read[i]);
        if(this->buffer_read[i]==33 && i==7){
            this->_typeOfReturn = 4;
        }
        if(this->buffer_read[i]==33 && i==2){
            this->_typeOfReturn = 5;
        }
    }
    if(this->_typeOfReturn == MSG_RETURN_POSITION){
        memcpy(this->_rcvPosition.encoded, this->buffer_read, sizeof this->_rcvPosition.encoded);

        _retInf[(this->_rcvPosition.decoded.id & 15)%3].id = (int) this->_rcvPosition.decoded.id;
        _retInf[(this->_rcvPosition.decoded.id & 15)%3].position = true;
        _retInf[(this->_rcvPosition.decoded.id & 15)%3].PosX = (int) this->_rcvPosition.decoded.curPosX;
        _retInf[(this->_rcvPosition.decoded.id & 15)%3].PosY = (int) this->_rcvPosition.decoded.curPosY;
        _retInf[(this->_rcvPosition.decoded.id & 15)%3].Angle = (float) this->_rcvPosition.decoded.curAngle/10;
        _retInf[(this->_rcvPosition.decoded.id & 15)%3].battery = false;
        _retInf[(this->_rcvPosition.decoded.id & 15)%3].BatteryLevel = 0;
        return this->_typeOfReturn;
    }
    else if(this->_typeOfReturn == MSG_RETURN_BATTERY){
        memcpy(this->_rcvBattery.encoded, this->buffer_read, sizeof this->_rcvBattery.encoded);

        _retInf[(this->_rcvBattery.decoded.id & 15)%3].id = 0;
        _retInf[(this->_rcvBattery.decoded.id & 15)%3].position = false;
        _retInf[(this->_rcvBattery.decoded.id & 15)%3].PosX = 0;
        _retInf[(this->_rcvBattery.decoded.id & 15)%3].PosY = 0;
        _retInf[(this->_rcvBattery.decoded.id & 15)%3].Angle = 0;
        _retInf[(this->_rcvBattery.decoded.id & 15)%3].battery = false;
        _retInf[(this->_rcvBattery.decoded.id & 15)%3].BatteryLevel = (float) this->_rcvBattery.decoded.batteryLevel/10;
        return this->_typeOfReturn;
    }
    return 0;
}

nrf_Communication::ReturnedInfo nrf_Communication::getInfoRet(int id){
    return _retInf[id%3];
}
