from collections import deque
from dataclasses import dataclass, field


@dataclass
class PController:
    Kp: float = 0.1

    def calc_input(self, sp: float, pv: float) -> float:
        """Calculate the control signal.
        sp: Set point
        pv: Process variable
        """
        e = sp - pv
        return self.Kp * e


@dataclass
class PIController:
    Kp: float = 0.1
    Ki: float = 0.01
    error_accumulator: float = 0

    def calc_input(self, sp: float, pv: float) -> float:
        """Calculate the control signal.
        sp: Set point
        pv: Process variable
        """
        e = sp - pv
        P = self.Kp * e

        self.error_accumulator += e
        I = self.Ki * self.error_accumulator

        return P + I


@dataclass
class PIDController:
    Kp: float = 0.1
    Ki: float = 0.01
    Kd: float = 0.1
    error_accumulator: float = 0
    prev_error: float = 0

    def calc_input(self, sp: float, pv: float, umin: float, umax: float) -> float:
        """Calculate the control signal.
        sp: Set point
        pv: Process variable
        """
        e = sp - pv
        P = self.Kp * e

        self.error_accumulator += e
        I = self.Ki * self.error_accumulator

        D = self.Kd * (e - self.prev_error)
        self.prev_error = e

        pid = P + I + D
        if pid < umin:
            return umin
        elif pid > umax:
            return umax
        else:
            return pid
