from time import sleep

import RPi.GPIO as GPIO


GPIO=3

GPIO.setmode(GPIO.BOARD)
GPIO.setup(GPIO, GPIO.OUT)
pwm=GPIO.PWM(GPIO, 50)
pwm.start(0)



def SetAngle(angle):

	duty = angle / 18 + 2

	GPIO.output(GPIO, True)

	pwm.ChangeDutyCycle(duty)

	sleep(1)

	GPIO.output(GPIO, False)

	pwm.ChangeDutyCycle(0)

SetAngle(90)
pwm.stop()
GPIO.cleanup()