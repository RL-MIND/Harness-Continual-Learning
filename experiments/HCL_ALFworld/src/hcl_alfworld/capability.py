from __future__ import annotations

from typing import List

from .schemas import CapabilityUnit


def default_capabilities() -> List[CapabilityUnit]:
    specifications = [
        (
            "navigation",
            "Navigate",
            ["go"],
            "move to a receptacle or room",
            ["destination is listed"],
            ["unknown destination"],
        ),
        (
            "inspection",
            "Inspect",
            ["look", "examine"],
            "observe a location or object",
            ["target is visible"],
            ["target is hidden"],
        ),
        (
            "open_close",
            "Open/close",
            ["open", "close"],
            "change a receptacle state",
            ["receptacle supports opening"],
            ["already open or closed"],
        ),
        (
            "pickup",
            "Pick up",
            ["take"],
            "put an accessible object in inventory",
            ["object is visible", "inventory has capacity"],
            ["object is enclosed or already held"],
        ),
        (
            "placement",
            "Place",
            ["put", "move"],
            "place an inventory object in/on a receptacle",
            ["object is held", "destination is accessible"],
            ["object is not held"],
        ),
        (
            "cleaning",
            "Clean",
            ["clean"],
            "clean an object at a sink",
            ["object is held", "sink is accessible"],
            ["wrong appliance"],
        ),
        (
            "heating",
            "Heat",
            ["heat"],
            "heat an object with a microwave",
            ["object is held", "microwave is accessible"],
            ["wrong appliance"],
        ),
        (
            "cooling",
            "Cool",
            ["cool"],
            "cool an object with a refrigerator",
            ["object is held", "refrigerator is accessible"],
            ["wrong appliance"],
        ),
        (
            "illumination",
            "Illuminate",
            ["use"],
            "use a lamp to illuminate an object",
            ["lamp is accessible"],
            ["wrong device"],
        ),
        ("inventory", "Inventory", ["inventory"], "inspect currently held objects", [], []),
    ]
    return [
        CapabilityUnit(
            capability_id=identifier,
            name=name,
            kind="atomic_external",
            function=function,
            action_prefixes=prefixes,
            input_output="ALFWorld text command -> environment observation",
            preconditions=preconditions,
            failure_modes=failures,
            version="alfworld-0.4.2",
            description=function,
            inputs=["current ALFWorld observation", "admissible commands"],
            outputs=["environment state transition and next observation"],
            success_conditions=["The environment accepts the grounded action."],
            confidence=1.0,
            status="validated",
            provider_component="alfworld_environment",
            model="environment-defined",
            prompt_version="not-applicable",
        )
        for identifier, name, prefixes, function, preconditions, failures in specifications
    ]
