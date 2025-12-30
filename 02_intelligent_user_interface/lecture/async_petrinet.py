import asyncio

from dataclasses import dataclass, field
from typing import Set, Callable, List, Optional


@dataclass
class Token:
    name: str = "default"
    value: str = "default"


@dataclass
class Place:
    name: str
    tokens: List[Token] = field(default_factory=list)

    def create_token(self, value: str = "default"):
        name = "default"
        self.tokens.append(Token(name, value))

    def remove_token(self):
        if len(self.tokens) > 0:
            self.tokens.pop()


@dataclass
class Transition:
    name: str
    input_places: List[Place]
    output_places: List[Place]
    observer_func: Optional[Callable] = None
    request_funcs: Set[Callable] = field(default_factory=set)


class PetriNet:
    def __init__(self):
        self.places = []
        self.transitions = []


class User:
    def __init__(self):
        self.observers: Set[Callable] = set()

    def attach(self, observer: Callable) -> None:
        self.observers.add(observer)

    async def respond_mood(self, mood: str) -> None:
        print("\n[User]: (Thinking about his mood...)\n")
        await asyncio.sleep(5)
        await self.notify(mood)
        print("\n[User]: Responded with the mood.\n")

    async def notify(self, mood: str) -> None:
        tasks = [asyncio.create_task(obs(mood)) for obs in self.observers]
        await asyncio.gather(*tasks)


class WeatherAPI:
    def __init__(self):
        self.observers: Set[Callable] = set()

    def attach(self, observer: Callable) -> None:
        self.observers.add(observer)

    async def respond_weather(self, weather: str) -> None:
        print("\n[WeatherAPI]: (Processing the request...)\n")
        await asyncio.sleep(1)
        await self.notify(weather)
        print("\n[WeatherAPI]: Responded with the weather.\n")

    async def notify(self, weather: str) -> None:
        tasks = [asyncio.create_task(obs(weather)) for obs in self.observers]
        await asyncio.gather(*tasks)


def transition_is_enabled(t: Transition) -> bool:
    """A transition is enabled if all input places have at least one token"""
    return all(len(place.tokens) > 0 for place in t.input_places)


async def fire_transition(t: Transition, value: str = "default", do_wait: bool = False) -> None:
    """Fire a transition by removing a token from each input place and adding a token to each output place.
    Then, fire request functions associated with the transition."""
    assert transition_is_enabled(t), "Transition is not enabled"

    for place in t.input_places:
        place.remove_token()
    for place in t.output_places:
        place.create_token(value=value)

    tasks = [asyncio.create_task(func()) for func in t.request_funcs]
    if tasks and do_wait:
        await asyncio.gather(*tasks)


async def fire_transitions(net: PetriNet) -> None:
    to_fire: List[Transition] = []
    for transition in net.transitions:
        # Skip transitions that are not enabled
        if not transition_is_enabled(transition):
            continue

        # If the transition has an observer function, skip it
        if transition.observer_func:
            continue
        to_fire.append(transition)

    for transition in to_fire:
        await fire_transition(transition)


def create_user_observer(t: Transition):
    async def user_observer(mood: str) -> None:
        if not transition_is_enabled(t):
            return
        print(f"User observer fired the transition {t.name}")
        await fire_transition(t, value=mood)

    return user_observer


def create_weather_observer(t: Transition):
    async def weather_observer(weather: str) -> None:
        if not transition_is_enabled(t):
            return
        print(f"Weather observer fired the transition {t.name}")
        await fire_transition(t, value=weather)

    return weather_observer


def print_petri_net(petrinet: PetriNet, print_transitions: bool = False) -> None:
    print("========================================")
    print("Places: ", end="")
    for place in petrinet.places:
        print(f"{place.name}: {len(place.tokens)}", end=", ")
    print()


async def main():
    # Create places
    p_init = Place(name="Init")
    p_req_user = Place(name="ReqUser")
    p_user_res_ready = Place(name="UserResReady")
    p_req_api = Place(name="ReqAPI")
    p_api_res_ready = Place(name="APIResReady")
    p_end = Place(name="End")

    # Create transitions
    t_start_conv = Transition("StartConv", [p_init], [p_req_user, p_req_api])
    t_res_user = Transition("ResUser", [p_req_user], [p_user_res_ready])
    t_res_api = Transition("ResAPI", [p_req_api], [p_api_res_ready])
    t_end_conv = Transition("EndConv", [p_user_res_ready, p_api_res_ready], [p_end])

    # Attach observer functions and requests
    t_res_user.observer_func = create_user_observer(t_res_user)
    t_res_api.observer_func = create_weather_observer(t_res_api)

    def request_user():
        print("\n[Agent] What is your mood now?")
        return user.respond_mood("chill")

    def request_api():
        print("\n[Agent] (Requesting the weather information)")
        return api.respond_weather("sunny")

    t_start_conv.request_funcs.add(request_user)
    t_start_conv.request_funcs.add(request_api)

    net = PetriNet()
    net.places = [p_init, p_req_user, p_user_res_ready, p_req_api, p_api_res_ready, p_end]
    net.transitions = [t_start_conv, t_res_user, t_res_api, t_end_conv]

    user = User()
    user.attach(t_res_user.observer_func)

    api = WeatherAPI()
    api.attach(t_res_api.observer_func)

    p_init.create_token()

    while True:
        # print_petri_net(net)
        await fire_transitions(net)

        await asyncio.sleep(1)

        if len(p_end.tokens) > 0:
            break
    print_petri_net(net)


if __name__ == "__main__":
    asyncio.run(main())
