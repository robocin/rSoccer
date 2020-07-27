#include "nrf.h"

NRF::NRF(std::string device, unsigned int baudrate) {
    //spdlog::get("Others")->info("Initializing NRF.");
    this->_device = device;
    this->_baudrate = baudrate;
}

NRF::~NRF() {
    //spdlog::get("Others")->info("Destroying NRF class");
}

char NRF::setup() {
    //spdlog::get("Others")->info("Setting up NRF with device {} and baudrate {}", this->_device, this->_baudrate);
    // Open an USB descriptor
    // O_RDWR: Open file for read and write
    // O_NOCTTY: If set and path identifies a terminal device, open() shall not cause the terminal device to become the controlling terminal for the process
    // O_NONBLOCK: If O_NONBLOCK is set, the open() function shall return without blocking for the device to be ready or available. Subsequent behavior of the device is device-specific.
    this->nrf_fd = open(this->_device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);

    // Clear terminal information structure
    memset (&this->_tty, 0, sizeof this->_tty);

    // Error Handling
    if (tcgetattr (this->nrf_fd, &this->_tty) != 0)
    {
        //spdlog::get("Others")->error("Unable to communicate with the serial port to do a get");
        return -1;
    }

    // Set Baud Rate
    cfsetospeed (&this->_tty, (speed_t)this->_baudrate);
    cfsetispeed (&this->_tty, (speed_t)this->_baudrate);


    // Setting other Port Stuff
    this->_tty.c_cflag     &=  ~PARENB;            // Make 8n1
    this->_tty.c_cflag     &=  ~CSTOPB;
    this->_tty.c_cflag     &=  ~CSIZE;
    this->_tty.c_cflag     |=  CS8;

    this->_tty.c_cflag     &=  ~CRTSCTS;           // no flow control
    this->_tty.c_cc[VMIN]   =  1;                  // read doesn't block
    this->_tty.c_cc[VTIME]  =  5;                  // 0.5 seconds read timeout
    this->_tty.c_cflag     |=  CREAD | CLOCAL;     // turn on READ & ignore ctrl lines

    // Make raw
    cfmakeraw(&this->_tty);


    // Flush Port, then applies attributes
    //tcsetattr(this->_USBDescriptor, TCSAFLUSH, &options);
    tcflush(this->nrf_fd, TCIFLUSH);
    if (tcsetattr (this->nrf_fd, TCSANOW, &this->_tty) != 0)
    {
        //spdlog::get("Others")->error("Unable to communicate with the serial port to do a set");
        return -2;
    }

    this->first_read = 1;
    return 0;
}

unsigned char NRF::packB1(){

    unsigned char b1 = 0;

    b1 = this->packet.b1.start;
    b1 = b1 << 6;
    b1 = b1 | (this->packet.b1.id << 4);
    b1 = b1 | (this->packet.b1.batteryRet << 3);
    b1 = b1 | (this->packet.b1.kickRet << 2);
    b1 = b1 | (this->packet.b1.motorSpeed1_p1);

    return b1;
}

unsigned char NRF::packB2(){

    unsigned char b2 = 0;

    b2 = this->packet.b2.motorSpeed1_p2;
    b2 = b2 << 2;
    b2 = b2 | (this->packet.b2.motorSpeed2_p1);

    return b2;
}

unsigned char NRF::packB3(){

    unsigned char b3 = 0;

    b3 = this->packet.b3.motorSpeed2_p2;
    b3 = b3 << 2;
    b3 = b3 | (packet.b3.stop);

    return b3;
}

int NRF::setMessage(int start, int id, int motorSpeed1, int motorSpeed2, int kickRet, int batteryRet, int stop){

    this->packet.b1.start = start;
    this->packet.b1.id = id;
    this->packet.b1.kickRet = kickRet;
    this->packet.b1.batteryRet = batteryRet;
    this->packet.b1.motorSpeed1_p1 = (motorSpeed1 & (0xC0)) >> 6; // 0xCO = 0B11000000

    this->packet.b2.motorSpeed1_p2 = (motorSpeed1 & (0x3F)); // 0x3F = 0B00111111
    this->packet.b2.motorSpeed2_p1 = (motorSpeed2 & (0xC0)) >> 6; // 0B11000000

    this->packet.b3.motorSpeed2_p2 = (motorSpeed2 & (0x3F)); // 0B00111111
    this->packet.b3.stop = stop;

    return 0;
}

void NRF::setRawBuffer(char b1, char b2, char b3){
    this->buffer_write[0] = b1;
    this->buffer_write[1] = b2;
    this->buffer_write[2] = b3;
    this->buffer_write[3] = 33; // End Tx char

}

int NRF::sendRawBuffer(){
    return write(this->nrf_fd, buffer_write, sizeof buffer_write );
}

int NRF::send(int start, int id, int motorSpeed1, int motorSpeed2, int kickRet, int batteryRet, int stop){
    if(this->_NRFNet == 1){
        this->setMessage(start, id + 3, motorSpeed1, motorSpeed2, kickRet, batteryRet, stop);
    }else{
        this->setMessage(start, id, motorSpeed1, motorSpeed2, kickRet, batteryRet, stop);
    }

    this->buffer_write[0] = this->packB1();
    this->buffer_write[1] = this->packB2();
    this->buffer_write[2] = this->packB3();
    this->buffer_write[3] = 33; // End Tx char
    int temp_return = -1;
    if(this->_shallSend){
        temp_return = write(this->nrf_fd, buffer_write, sizeof buffer_write );
    }
    return temp_return;
}

int NRF::send(int id, int motorSpeed1, int motorSpeed2, int kickRet, int batteryRet){
    return send(_MSG_BEGIN, id, motorSpeed1, motorSpeed2, kickRet, batteryRet, _MSG_END);
}

int NRF::send(int id, std::pair<double, double> speed, int kickRet, int batteryRet){
    return send(_MSG_BEGIN, id, (int)speed.first, (int)speed.second, kickRet, batteryRet, _MSG_END);
}

void NRF::recv(){
    ssize_t var =read(this->nrf_fd, buffer_read, sizeof buffer_read);
    if(var){} // do something;
    //printf("read: %d bytes\n", r);

    if(this->first_read){
        memset(&this->buffer_read, 0, sizeof buffer_read);
        this->first_read = 0;
    }

    /*int i = 0;
    while(buffer_read[i] != 33 && !first_read){
        printf("%d|", buffer_read[i++] );
    }*/

   for(int i = 0; i < 64; i++){
        printf("%d|",buffer_read[i] );
        //if(buffer_read[i] == 33) break;
    }

    memset(&this->buffer_read, 0, sizeof buffer_read);

}

void NRF::startSending() {
    this->_shallSend = true;
}

void NRF::stopSending() {
    this->_shallSend = false;
}

void NRF::setNet(int ChoosenNet) {
    this->_NRFNet = ChoosenNet;
}
