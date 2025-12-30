from dataclasses import dataclass
from typing import List


@dataclass
class Place:
    name: str
    token_count: int = 0


@dataclass
class Transition:
    name: str
    input_places: List[Place]
    output_places: List[Place]


class PetriNet:
    def __init__(self):
        self.places = []
        self.transitions = []


def create_transition(
        name: str,
        input_places: List[Place],
        output_places: List[Place]
) -> Transition:
    return Transition(name, input_places, output_places)


def transition_is_enabled(t: Transition) -> bool:
    return all(place.token_count > 0 for place in t.input_places)


def fire_transition(t: Transition) -> None:
    assert transition_is_enabled(t), "Transition is not enabled"

    for place in t.input_places:
        place.token_count -= 1
    for place in t.output_places:
        place.token_count += 1


def fire_transitions(net: PetriNet) -> None:
    to_fire: List[Transition] = []
    for transition in net.transitions:
        if transition_is_enabled(transition):
            to_fire.append(transition)

    for transition in to_fire:
        fire_transition(transition)


def print_petri_net(petrinet: PetriNet, print_transitions: bool = False) -> None:
    print("========================================")
    print("Places:")
    print("  ", end="")
    for place in petrinet.places:
        print(f"{place.name}: {place.token_count}", end=", ")
    print()

    if not print_transitions:
        return

    print("Transitions:")
    for transition in petrinet.transitions:
        print(f"  {transition.name}")
        print("    Input places: ", end="")
        for place in transition.input_places:
            print(f"{place.name}", end=", ")
        print()
        print("    Output places: ", end="")
        for place in transition.output_places:
            print(f"{place.name}", end=", ")
        print()
    print()

    return


def main() -> None:
    # Create a simple Petri net
    net = PetriNet()
    p_init = Place(name="Init", token_count=1)
    p_req_user = Place(name="ReqUser")
    p_user_res_ready = Place(name="UserResReady")
    p_req_api = Place(name="ReqAPI")
    p_api_res_ready = Place(name="APIResReady")
    p_end = Place(name="End")

    t_start_conv = create_transition("StartConv", [p_init], [p_req_user, p_req_api])
    t_res_user = create_transition("ResUser", [p_req_user], [p_user_res_ready])
    t_res_api = create_transition("ResAPI", [p_req_api], [p_api_res_ready])
    t_combine_res = create_transition("CombineRes", [p_user_res_ready, p_api_res_ready], [p_end])

    net.places.extend([p_init, p_req_user, p_user_res_ready, p_req_api, p_api_res_ready, p_end])
    net.transitions.extend([t_start_conv, t_res_user, t_res_api, t_combine_res])

    # Run the Petri net and print the state
    print_petri_net(net)

    fire_transitions(net)
    print_petri_net(net)

    fire_transitions(net)
    print_petri_net(net)

    fire_transitions(net)
    print_petri_net(net)

    fire_transitions(net)
    print_petri_net(net)


if __name__ == "__main__":
    main()
