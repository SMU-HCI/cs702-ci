#!/usr/bin/env python3
"""Model-check a PRISM model and property file with Stormpy."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import stormpy


def _strip_comments(text: str) -> str:
    """Remove PRISM // and /* */ comments while preserving quoted strings."""
    output: list[str] = []
    index = 0
    in_string = False

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            output.append(char)
            if char == "\\" and next_char:
                output.append(next_char)
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
        elif char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] != "\n":
                index += 1
        elif char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and text[index : index + 2] != "*/":
                if text[index] == "\n":
                    output.append("\n")
                index += 1
            if index + 1 >= len(text):
                raise ValueError("Unterminated block comment in property input")
            index += 2
        else:
            output.append(char)
            index += 1

    return "".join(output)


def _split_properties(text: str) -> list[str]:
    """Split a PRISM property file whose properties may omit semicolons."""
    text = _strip_comments(text)
    properties: list[str] = []
    current: list[str] = []
    square_depth = 0
    round_depth = 0
    curly_depth = 0
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            current.append(char)
        elif char == "[":
            square_depth += 1
            current.append(char)
        elif char == "]":
            square_depth -= 1
            if square_depth < 0:
                raise ValueError("Unbalanced delimiters in property input")
            current.append(char)
        elif char == "(":
            round_depth += 1
            current.append(char)
        elif char == ")":
            round_depth -= 1
            if round_depth < 0:
                raise ValueError("Unbalanced delimiters in property input")
            current.append(char)
        elif char == "{":
            curly_depth += 1
            current.append(char)
        elif char == "}":
            curly_depth -= 1
            if curly_depth < 0:
                raise ValueError("Unbalanced delimiters in property input")
            current.append(char)
        elif (
            char in {";", "\n"}
            and square_depth == 0
            and round_depth == 0
            and curly_depth == 0
        ):
            statement = "".join(current).strip()
            if statement:
                properties.append(statement)
            current = []
        else:
            current.append(char)

    statement = "".join(current).strip()
    if statement:
        properties.append(statement)

    if in_string:
        raise ValueError("Unbalanced delimiters in property input")
    if square_depth or round_depth or curly_depth:
        raise ValueError("Unbalanced delimiters in property input")
    return properties


_BOUNDED_GLOBALLY = re.compile(
    r"""
    ^
    (?P<prefix>
        \s*P(?:min|max)?\s*
        (?:=\s*\?|(?:<=|>=|<|>)\s*[^\[]+)
        \[\s*
    )
    G\s*<=\s*(?P<bound>[^\s]+)\s+
    (?P<operand>.+?)
    (?P<suffix>\s*\]\s*)
    $
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _make_storm_compatible(statement: str) -> str:
    """Translate PRISM syntax not accepted by Storm's property parser."""
    match = _BOUNDED_GLOBALLY.match(statement)
    if match is None:
        return statement

    # G<=k phi is equivalent to !(F<=k !phi).
    return (
        f"{match.group('prefix')}!(F<={match.group('bound')} "
        f"!({match.group('operand')})){match.group('suffix')}"
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stormpy on an existing PRISM .pm model and .pctl file."
    )
    parser.add_argument("model", type=Path, help="PRISM model file (.pm)")
    parser.add_argument(
        "properties",
        nargs="?",
        type=Path,
        help="PRISM property file (.pctl)",
    )
    parser.add_argument(
        "-p",
        "--property",
        dest="property_text",
        help='check one inline property, e.g. \'P=? [ F "converted" ]\'',
    )
    arguments = parser.parse_args()

    if (arguments.properties is None) == (arguments.property_text is None):
        parser.error("provide either a property file or --property (but not both)")
    return arguments


def _format_result(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def run(model_path: Path, property_text: str) -> int:
    statements = _split_properties(property_text)
    if not statements:
        raise ValueError("No properties found")

    prism_program = stormpy.parse_prism_program(str(model_path))
    properties = []
    for statement in statements:
        compatible_statement = _make_storm_compatible(statement)
        parsed = stormpy.parse_properties_for_prism_program(
            compatible_statement, prism_program
        )
        if len(parsed) != 1:
            raise ValueError(f"Expected one property, got {len(parsed)}: {statement}")
        properties.append(parsed[0])

    model = stormpy.build_model(
        prism_program, [prop.raw_formula for prop in properties]
    )
    initial_states = list(model.initial_states)
    if not initial_states:
        raise ValueError("The model has no initial state")

    print(
        f"Model: {model_path} "
        f"({model.model_type.name}, {model.nr_states} states, "
        f"{model.nr_transitions} transitions)"
    )
    print(f"Properties: {len(properties)}")

    for number, (statement, prop) in enumerate(
        zip(statements, properties, strict=True), start=1
    ):
        result = stormpy.model_checking(model, prop)
        print(f"\n({number}) {statement}")
        if len(initial_states) == 1:
            print(f"Result: {_format_result(result.at(initial_states[0]))}")
        else:
            for state in initial_states:
                print(
                    f"Result at initial state {state}: "
                    f"{_format_result(result.at(state))}"
                )

    return 0


def main() -> int:
    arguments = _parse_arguments()
    try:
        if arguments.properties is not None:
            property_text = arguments.properties.read_text(encoding="utf-8")
        else:
            property_text = arguments.property_text
        return run(arguments.model, property_text)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
