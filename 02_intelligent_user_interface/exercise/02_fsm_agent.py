"""
A sample implementation of a finite state machine (FSM).
"""
from abc import ABC, abstractmethod
from collections import namedtuple


Event = namedtuple("Event", ["type", "value"])
Output = namedtuple("Output", ["state", "value"])
Transition = namedtuple("Transition", ["source", "target", "event"])


class State(ABC):
    @abstractmethod
    def next(self, event: Event) -> Transition:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> Output:
        raise NotImplementedError


class WeatherState(State):
    def next(self, event: Event) -> Transition:
        if event.type == "Weather" and event.value in ["sunny", "rainy"]:
            return Transition(self, MoodState(), event)
        else:
            # Implement this part!!!
            raise NotImplementedError("Implement a transition to AskRainingState")


    def run(self) -> Output:
        return Output(self, f"Running WeatherState")

    def __str__(self):
        return "WeatherState"


class AskRainingState(State):
    def next(self, event: Event) -> Transition:
        if event.type == "YesNo" and event.value in ["yes", "no"]:
            # Implement this part!!!
            raise NotImplementedError("Implement a transition to MoodState")
        else:
            # Implement this part!!!
            raise NotImplementedError("Implement a transition to AskRainingState")

    def run(self) -> Output:
        return Output(self, f"Running AskRainingState")

    def __str__(self):
        return "AskRainingState"


class MoodState(State):
    def next(self, event: Event) -> Transition:
        if event.type == "Mood" and event.value in ["active", "chill"]:
            return Transition(self, TerminalState(), event)
        else:
            return Transition(self, self, None)

    def run(self) -> Output:
        return Output(self, "Running MoodState")

    def __str__(self):
        return "MoodState"


class TerminalState(State):
    def next(self, event: Event) -> Transition:
        return Transition(self, self, None)

    def run(self) -> Output:
        return Output(self, "Running TerminalState")

    def __str__(self):
        return "TerminalState"


class StateMachine:
    def __init__(self):
        print(f"Move to a WeatherState\n")
        self.state = WeatherState()
        self.parameters = {}

    def handle(self, event: Event) -> None:
        # Run the state and print the output.
        output = self.state.run()
        print(f"State {output.state} emits {output.value}")

        # Handle the event and perform the transition.
        print(f"Received an event of type {event.type} with value {event.value}")
        transition = self.state.next(event)

        if transition.event is not None:
            print(f"Transition from {transition.source} to {transition.target}")
            # Implement this part!!!
            raise NotImplementedError("Handle the YesNo event appropriately")
            self.parameters[transition.event.type] = transition.event.value

        self.state = transition.target

        # If FSM reaches TerminalState and we have all the weather parameters,
        # call the `get_weather` function and reset the weather parameters.
        if isinstance(transition.target, TerminalState):
            weather = self.parameters["Weather"]
            mood = self.parameters["Mood"]
            recommend_activity(weather, mood)

        print()


def recommend_activity(weather: str, mood: str) -> None:
    """Recommend an activity based on the weather and the mood."""
    match (weather, mood):
        case ("sunny", "active"):
            print("You should go cycling!")
        case ("sunny", "chill"):
            print("You should go on a picnic!")
        case ("rainy", "active"):
            print("You should go indoor climbing!")
        case ("rainy", "chill"):
            print("You should watch a movie!")
        case _:
            raise ValueError(f"Invalid weather and/or mood: {weather}, {mood}")


if __name__ == "__main__":
    sm = StateMachine()

    while True:
        if isinstance(sm.state, WeatherState):
            event_value = input("Enter the weather (sunny | rainy): ")
            input_event = Event("Weather", event_value)
        elif isinstance(sm.state, MoodState):
            event_value = input("Enter the mood (active | chill): ")
            input_event = Event("Mood", event_value)
        elif isinstance(sm.state, AskRainingState):
            # Implement this part!!!
            raise NotImplementedError("Handle the YesNo event")
        elif isinstance(sm.state, TerminalState):
            break
        else:
            raise ValueError(f"Invalid state: {sm.state}")
        sm.handle(input_event)

