"""
A sample implementation of a frame-based agent.
"""
import copy

from dataclasses import dataclass
from enum import Enum


class Weather(Enum):
    RAINY = "rainy"
    SUNNY = "sunny"


class Mood(Enum):
    ACTIVE = "active"
    CHILL = "chill"


@dataclass
class Frame:
    weather: Weather = None
    mood: Mood = None


@dataclass
class InputEvent:
    weather: Weather = None
    mood: Mood = None


class Action(Enum):
    ASK_ALL = "ask all"
    ASK_WEATHER = "ask weather"
    ASK_MOOD = "ask mood"
    RECOMMEND_ACTIVITY = "recommend activity"


def update_frame(input_event: InputEvent, frame: Frame) -> Frame:
    """Process the input and update the frame."""
    frame = copy.deepcopy(frame)
    if input_event.weather is not None:
        frame.weather = input_event.weather
    if input_event.mood is not None:
        frame.mood = input_event.mood
    return frame


def determine_action(frame: Frame) -> Action:
    """Check if the frame is fully defined."""
    if frame.weather is None and frame.mood is None:
        return Action.ASK_ALL
    elif frame.weather is None:
        return Action.ASK_WEATHER
    elif frame.mood is None:
        return Action.ASK_MOOD
    else:
        return Action.RECOMMEND_ACTIVITY


def recommend_activity(frame: Frame) -> None:
    """Recommend an activity based on the frame."""
    match frame:
        case Frame(weather="sunny", mood="active"):
            print("You should go cycling!")
        case Frame(weather="sunny", mood="chill"):
            print("You should go on a picnic!")
        case Frame(weather="rainy", mood="active"):
            print("You should go indoor climbing!")
        case Frame(weather="rainy", mood="chill"):
            print("You should watch a movie!")
        case _:
            raise ValueError(f"Frame is not fully specified: {frame}")


if __name__ == "__main__":
    current_frame = Frame()
    while True:
        # Determine the action based on the frame.
        action = determine_action(current_frame)

        # Perform the action.
        # If the action is `recommend activity`, recommend an activity and break the loop.
        if action == Action.RECOMMEND_ACTIVITY:
            recommend_activity(current_frame)
            break

        # Otherwise, ask the user for input.
        if action == Action.ASK_ALL:
            response = input("What's the weather like? What's your mood? (weather:rainy|sunny; mood:active|chill) ")
        elif action == Action.ASK_WEATHER:
            response = input("What's the weather like? (weather:rainy|sunny) ")
        elif action == Action.ASK_MOOD:
            response = input("What's your mood? (mood:active|chill) ")
        else:
            raise ValueError(f"Unknown action: {action}")

        # Parse the response and update the frame.
        items = response.split(";")
        event = InputEvent()
        for item in items:
            try:
                key, value = item.split(":")
                key = key.strip()
                value = value.strip()
                if key == "weather" and value in (item.value for item in Weather):
                    event.weather = value
                elif key == "mood" and value in (item.value for item in Mood):
                    event.mood = value
            except ValueError:
                pass

        current_frame = update_frame(event, current_frame)
        print(f"Current Frame: {current_frame}\n")
