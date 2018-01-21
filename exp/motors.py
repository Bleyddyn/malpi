#!/usr/bin/python

from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor

class Motors():
    
    def __init__( self, i2c_address=0x60 ):
        self.speed = 150
        self.last_command = None
        self.mh = Adafruit_MotorHAT(addr=i2c_address)

    def __enter__(self):
        """ Allow the use of 'with Motors() as mot:'
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def command(self, command):
        if command.startswith( 'forward', 0, len('forward') ):
            self.driveForward(self.speed)
        elif command.startswith( 'backward', 0, len('backward') ):
            self.driveBackward(self.speed)
        elif command.startswith( 'left', 0, len('left') ):
            self.turnLeft(self.speed)
        elif command.startswith( 'right', 0, len('right') ):
            self.turnRight(self.speed)
        elif command.startswith( 'stop', 0, len('stop') ):
            self.stop()
        elif command.startswith( 'speed', 0, len('speed') ):
            self.setSpeed( line[len('speed '):] )
        else:
            raise ValueError( "Invalid motor command: " + str(command) )

    def setSpeed( self, newSpeed ):
        spInt = int(newSpeed)
        if spInt >= 0 and spInt < 256 and spInt != self.speed:
            self.speed = spInt
            if self.last_command:
                self.last_command(spInt)
        else:
            raise ValueError("motor speed must be between 0 adn 255")

    def setMotor( self, mnum, forward, speed ):
        myMotor = self.mh.getMotor(mnum)

        # set the speed to use, from 0 (off) to 255 (max speed)
        myMotor.setSpeed(speed)
        if forward:
            myMotor.run(Adafruit_MotorHAT.FORWARD)
        else:
            myMotor.run(Adafruit_MotorHAT.BACKWARD)

    def driveForward( self, speed=150 ):
        self.last_command = self.driveForward
        self.setMotor(1, True, speed)
        self.setMotor(2, True, speed)
        self.setMotor(3, True, speed)
        self.setMotor(4, True, speed)

    def driveBackward( self, speed=150 ):
        self.last_command = self.driveBackward
        self.setMotor(1, False, speed)
        self.setMotor(2, False, speed)
        self.setMotor(3, False, speed)
        self.setMotor(4, False, speed)

    def turnLeft( self, speed=150 ):
        self.last_command = self.turnLeft
        self.setMotor(1, False, speed)
        self.setMotor(2, False, speed)
        self.setMotor(3, True, speed)
        self.setMotor(4, True, speed)

    def turnRight( self, speed=150 ):
        self.last_command = self.turnRight
        self.setMotor(1, True, speed)
        self.setMotor(2, True, speed)
        self.setMotor(3, False, speed)
        self.setMotor(4, False, speed)

    def stop(self):
        self.last_command = None
        self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

if __name__ == "__main__":
    tests = ['forward', 'left', 'right', 'backward', 'stop']
    with Motors() as motor:
        for t in tests:
            print t
            motor.command(t)

        try:
            motor.command('albkjsladkjf')
        except ValueError as err:
            print err
            
