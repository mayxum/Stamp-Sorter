# Homes the carousel exactly like sorter_app.py, then lets you nudge it either way until
# bin 0 sits centered under the trapdoor. Prints HOME_OFFSET_STEPS for sorter_app.py.
# Run on the Pi with the system Python (sorter_app.py must NOT be running):
#     python3 ~/home_offset.py

import time

from gpiozero import DigitalInputDevice, OutputDevice

STEP_PIN, DIR_PIN, EN_PIN, HOME_PIN = 17, 27, 22, 23
HOMING_DELAY = 0.01
MAX_HOMING_STEPS = 400

step = OutputDevice(STEP_PIN)
direction = OutputDevice(DIR_PIN)
enable = OutputDevice(EN_PIN, active_high=False, initial_value=False)
home = DigitalInputDevice(HOME_PIN, pull_up=None, active_state=True)


def pulse(count):
    for _ in range(count):
        step.on()
        time.sleep(HOMING_DELAY)
        step.off()
        time.sleep(HOMING_DELAY)


offset = 0
try:
    enable.on()
    direction.on()  # same direction as sorter_app.py homing
    steps = 0
    while home.value:  # already in the slot: move out first
        pulse(1)
        steps += 1
        if steps >= MAX_HOMING_STEPS:
            raise SystemExit("Sensor reads blocked all the time: check the wiring.")
    while not home.value:
        pulse(1)
        steps += 1
        if steps >= MAX_HOMING_STEPS:
            raise SystemExit(
                "Home not found after 2 turns: check the wiring and that the flag passes through the slot."
            )
    print("Home found. Now move bin 0 under the trapdoor.")
    print(
        "Type steps to move: positive = forward, negative = back (1 step = 1.8 degrees, 40 = one bin). Enter 0 when centered."
    )
    while True:
        text = input("steps> ").strip()
        try:
            k = int(text or 0)
        except ValueError:
            print("Type a whole number.")
            continue
        if k == 0:
            break
        if k < 0:
            direction.off()
            pulse(-k)
            direction.on()
        else:
            pulse(k)
        offset += k
        print("offset so far:", offset)
finally:
    enable.off()

print("\nHOME_OFFSET_STEPS = %d" % offset)
