echo "sudo touch /etc/udev/rules.d/99-usb-serial.rules"
sudo rm /etc/udev/rules.d/99-usb-serial.rules
sudo touch /etc/udev/rules.d/99-usb-serial.rules
# FINDING ATTR OF USB: udevadm info --attribute-walk /dev/ttyACM1
#FTDIs
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6001\", ATTRS{serial}==\"AI05BA5R\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6001\", ATTRS{serial}==\"AI05BAVI\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6015\", ATTRS{serial}==\"DA01OEX3\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
#F7s
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0483\", ATTRS{idProduct}==\"374b\", ATTRS{serial}==\"066FFF535155878281123442\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0483\", ATTRS{idProduct}==\"374b\", ATTRS{serial}==\"066EFF535155878281124446\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
#H7s
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0483\", ATTRS{idProduct}==\"374e\", ATTRS{serial}==\"005400383137511739383538\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0483\", ATTRS{idProduct}==\"374e\", ATTRS{serial}==\"003C00343137510F39383538\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0483\", ATTRS{idProduct}==\"374e\", ATTRS{serial}==\"0017003B3137510F39383538\", SYMLINK+=\"NRF_USB\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules

# OLD VERSION OF RADIO (XBEE)
# echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6001\", ATTRS{serial}==\"AH01GYV4\", SYMLINK+=\"xBeeMaster\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
# echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6015\", ATTRS{serial}==\"DN01E5ZP\", SYMLINK+=\"xBeeMaster\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules
# echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6015\", ATTRS{serial}==\"DN01E5ZT\", SYMLINK+=\"xBeeMaster\"" | sudo tee --append /etc/udev/rules.d/99-usb-serial.rules

echo "sudo udevadm control --reload-rules"
sudo udevadm control --reload-rules

sudo usermod -a -G dialout $USER