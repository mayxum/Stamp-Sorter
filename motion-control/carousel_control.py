import os

SIMULATE = os.environ.get("SORTER_SIM") == "1"
if SIMULATE:
    # Must be set before gpiozero is imported.
    os.environ["GPIOZERO_PIN_FACTORY"] = "mock"
    os.environ["GPIOZERO_MOCK_PIN_CLASS"] = "mockpwmpin"

import time
from gpiozero import OutputDevice, DigitalInputDevice, AngularServo

# Pin assignments (BCM numbering), matching interface PCB J1
STEP_PIN = 17   # Pi pin 11 -> J1 STEP -> A4988 STEP
DIR_PIN = 27    # Pi pin 13 -> J1 DIR  -> A4988 DIR
EN_PIN = 22     # Pi pin 15 -> J1 EN   -> A4988 EN (active low, R3 pull-up)
HOME_PIN = 23   # Pi pin 16 -> J1 HOME <- LTH-301-05 collector (R2 pull-up)
SERVO_PIN = 18  # Pi pin 12 -> J1 PWM  -> SG90 signal (via J5)

# Carousel mechanics
TOTAL_BINS = 5
STEPS_PER_REV = 200                           # 1.8 degree motor, full step (MS1-MS3 tied to GND)
STEPS_PER_BIN = STEPS_PER_REV // TOTAL_BINS   # 40 steps per bin
STEP_DELAY = 0.003                            # seconds high and low per step
HOMING_DELAY = 0.01                           # slower stepping while homing
MAX_HOMING_STEPS = 2 * STEPS_PER_REV          # give up after two full turns

# Outputs
step_out = OutputDevice(STEP_PIN)
dir_out = OutputDevice(DIR_PIN)
# active_high=False: on() drives the pin LOW (driver enabled),
# off() drives it HIGH (driver disabled). Start disabled.
enable_out = OutputDevice(EN_PIN, active_high=False, initial_value=False)

# Home sensor: external pull-up R2, so no internal pull.
# value is True when HOME is high, which means the beam is blocked by the flag.
home_in = DigitalInputDevice(HOME_PIN, pull_up=None, active_state=True)

# SG90 trapdoor servo (50 Hz PWM on GPIO 18).
# Pulse widths from the original code. If the servo buzzes at 0 degrees,
# raise min_pulse_width slightly (for example 0.0006).
trapdoor_servo = AngularServo(
    SERVO_PIN,
    min_angle=0,
    max_angle=180,
    min_pulse_width=0.0005,
    max_pulse_width=0.0025,
    initial_angle=0,          # trapdoor closed at startup
)

current_bin = 0


def pulse_steps(count, delay):
    """Send count step pulses to the A4988."""
    for _ in range(count):
        step_out.on()
        time.sleep(delay)
        step_out.off()
        time.sleep(delay)


def home_carousel():
    """Turn the carousel until the flag blocks the home sensor, then call that bin 0."""
    global current_bin
    print("Homing carousel...")
    trapdoor_servo.angle = 0
    enable_out.on()          # enable the driver (EN low)
    dir_out.on()             # clockwise (actual direction depends on motor wiring)

    steps = 0
    while not home_in.value:             # HOME low = beam clear, keep turning
        if steps >= MAX_HOMING_STEPS:
            enable_out.off()
            raise RuntimeError(
                "Home sensor never triggered. Check the sensor wiring and the flag."
            )
        pulse_steps(1, HOMING_DELAY)
        steps += 1
        if SIMULATE and steps >= 5:
            home_in.pin.drive_high()     # simulate the flag blocking the beam
    if SIMULATE:
        home_in.pin.drive_low()          # flag moves away again in the simulation

    current_bin = 0
    print(f"Home found after {steps} steps. Carousel at bin 0.\n")


def rotate_to_bin(target_bin):
    """Move to target_bin the shortest way around."""
    global current_bin
    if target_bin == current_bin:
        print(f"Already at bin {current_bin}.")
        return

    diff = (target_bin - current_bin) % TOTAL_BINS    # 0 to 4

    if diff <= 2:
        dir_out.on()                                  # clockwise
        steps_to_move = diff * STEPS_PER_BIN
        direction = "clockwise"
    else:
        dir_out.off()                                 # counterclockwise
        steps_to_move = (TOTAL_BINS - diff) * STEPS_PER_BIN
        direction = "counterclockwise"

    print(f"Moving {steps_to_move} steps {direction}.")
    pulse_steps(steps_to_move, STEP_DELAY)
    current_bin = target_bin
    print(f"At bin {current_bin}.")


def get_vision_target():
    """Placeholder for the vision result: ask for a bin number 0 to 4."""
    while True:
        user_input = input("\nTarget bin (0-4) or 'q' to quit: ").strip().lower()
        if user_input == "q":
            raise KeyboardInterrupt
        if user_input.isdigit() and 0 <= int(user_input) < TOTAL_BINS:
            return int(user_input)
        print("Enter a number from 0 to 4.")


def trigger_trapdoor():
    """Open the trapdoor 90 degrees to drop the stamp, then close it."""
    trapdoor_servo.angle = 90
    print("Trapdoor open.")
    time.sleep(0.65)          # about 0.18 s servo travel plus time for the stamp to drop
    trapdoor_servo.angle = 0
    print("Trapdoor closed.")
    time.sleep(0.20)          # let it settle before the carousel moves again


if __name__ == "__main__":
    print("Mode:", "SIMULATION (mock pins)" if SIMULATE else "HARDWARE")
    try:
        home_carousel()
        while True:
            target = get_vision_target()
            rotate_to_bin(target)
            trigger_trapdoor()
    except KeyboardInterrupt:
        pass
    finally:
        trapdoor_servo.angle = 0
        enable_out.off()      # disable the driver (EN high) so the motor is not powered at rest
        for device in (step_out, dir_out, enable_out, home_in, trapdoor_servo):
            device.close()
        print("\nDriver disabled and GPIO pins released.")
