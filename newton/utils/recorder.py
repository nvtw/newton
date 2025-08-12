# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import pickle

import numpy as np
import warp as wp

from newton.sim.model import Model
from newton.sim.state import State

from collections.abc import Mapping, Iterable
import json
import numpy as np


def serialize_ndarray(arr: np.ndarray) -> dict:
    """
    Serialize a numpy ndarray to a dictionary representation.

    Args:
        arr: The numpy array to serialize.

    Returns:
        A dictionary containing the array's type, dtype, shape, and data.
    """
    return {"__type__": "numpy.ndarray", "dtype": str(arr.dtype), "shape": arr.shape, "data": json.dumps(arr.tolist())}


def deserialize_ndarray(data: dict) -> np.ndarray:
    """
    Deserialize a dictionary representation back to a numpy ndarray.

    Args:
        data: Dictionary containing the serialized array data.

    Returns:
        The reconstructed numpy array.
    """
    if data.get("__type__") != "numpy.ndarray":
        raise ValueError("Invalid data format for numpy array deserialization")

    dtype = np.dtype(data["dtype"])
    shape = tuple(data["shape"])
    array_data = json.loads(data["data"])

    return np.array(array_data, dtype=dtype).reshape(shape)


def serialize(obj, callback, _visited=None, _path=""):
    """
    Recursively serialize an object into a dict, handling primitives,
    containers, and custom class instances. Calls callback(obj) for every object
    and replaces obj with the callback's return value before continuing.

    Args:
        obj: The object to serialize.
        callback: A function taking two arguments (the object and current path) and returning the (possibly transformed) object.
        _visited: Internal set to avoid infinite recursion from circular references.
        _path: Internal parameter tracking the current path/member name.
    """
    if _visited is None:
        _visited = set()

    # Run through callback first (object may be replaced)
    result = callback(obj, _path)
    if result is not obj:
        return result

    obj_id = id(obj)
    if obj_id in _visited:
        return "<circular_reference>"

    # Add to visited set (stack-like behavior)
    _visited.add(obj_id)

    try:
        # Primitive types
        if isinstance(obj, (str, int, float, bool, type(None))):
            return {"__type__": type(obj).__name__, "value": obj}

        # NumPy scalar types
        if isinstance(obj, np.number):
            return {
                "__type__": type(obj).__name__,
                "value": obj.item(),  # Convert numpy scalar to Python scalar
            }

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return serialize_ndarray(obj)

        # Mappings (like dict)
        if isinstance(obj, Mapping):
            return {
                "__type__": type(obj).__name__,
                "items": {
                    str(k): serialize(v, callback, _visited, f"{_path}.{k}" if _path else str(k))
                    for k, v in obj.items()
                },
            }

        # Iterables (like list, tuple, set)
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
            return {
                "__type__": type(obj).__name__,
                "items": [
                    serialize(item, callback, _visited, f"{_path}[{i}]" if _path else f"[{i}]")
                    for i, item in enumerate(obj)
                ],
            }

        # Custom object — serialize attributes
        if hasattr(obj, "__dict__"):
            return {
                "__type__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "attributes": {
                    attr: serialize(value, callback, _visited, f"{_path}.{attr}" if _path else attr)
                    for attr, value in vars(obj).items()
                },
            }

        # Fallback — non-serializable type
        raise ValueError(f"Cannot serialize object of type {type(obj)}")
    finally:
        # Remove from visited set when done (stack-like cleanup)
        _visited.discard(obj_id)


def serialize_newton(obj):
    def callback(x, path):        
        if isinstance(x, wp.array):
            # print(f"serialize warp.array at path: {path}")
            return {"__type__": "warp.array", "data": serialize_ndarray(x.numpy())}

        if isinstance(x, wp.HashGrid):
            # print(f"serialize warp.types.HashGrid at path: {path}")
            return {"__type__": "warp.HashGrid", "data": None}

        if isinstance(x, wp.Mesh):
            # print(f"serialize warp.Mesh at path: {path}")
            return {"__type__": "warp.Mesh", "data": None}

        if isinstance(x, wp.context.Device):
            return {"__type__": "wp.context.Device", "data": None}        

        if callable(x):
            # print(f"serialize callable at path: {path}")
            return {"__type__": "callable", "data": None}

        print(type(x))

        return x

    return serialize(obj, callback)


def transfer_to_model(source_dict, target_obj, callback=None, _path=""):
    """
    Recursively transfer values from source_dict to target_obj, respecting the tree structure.
    Only transfers values where both source and target have matching attributes.

    Args:
        source_dict: Dictionary containing the values to transfer (from deserialization).
        target_obj: Target object to receive the values.
        callback: Optional function taking (value, path) and returning transformed value.
        _path: Internal parameter tracking the current path.
    """
    if not hasattr(target_obj, "__dict__"):
        return

    # Handle case where source_dict is not a dict (primitive value)
    if not isinstance(source_dict, dict):
        return

    # Iterate through all attributes of the target object
    for attr_name in dir(target_obj):
        # Skip private/magic methods and properties
        if attr_name.startswith("_"):
            continue

        # Skip if attribute doesn't exist in target or is not settable
        if not hasattr(target_obj, attr_name):
            continue

        try:
            target_value = getattr(target_obj, attr_name)
        except (AttributeError, TypeError):
            # Skip attributes that can't be accessed
            continue

        # Skip methods and non-data attributes
        if callable(target_value):
            continue

        # Check if source_dict has this attribute
        if attr_name not in source_dict:
            continue

        source_value = source_dict[attr_name]
        current_path = f"{_path}.{attr_name}" if _path else attr_name

        # Apply callback if provided
        if callback is not None:
            transformed_value = callback(source_value, current_path)
            if transformed_value is not source_value:
                source_value = transformed_value

        # Handle different types of values
        if hasattr(target_value, "__dict__") and isinstance(source_value, dict):
            # Recursively transfer for custom objects
            transfer_to_model(source_value, target_value, callback, current_path)
        elif isinstance(source_value, (list, tuple)) and hasattr(target_value, "__len__"):
            # Handle sequences - try to transfer if lengths match or target is empty
            try:
                if len(target_value) == 0 or len(target_value) == len(source_value):
                    # For now, just assign the value directly
                    # In a more sophisticated implementation, you might want to handle
                    # element-wise transfer for lists of objects
                    setattr(target_obj, attr_name, source_value)
            except (TypeError, AttributeError):
                # If we can't handle the sequence, try direct assignment
                try:
                    setattr(target_obj, attr_name, source_value)
                except (AttributeError, TypeError):
                    # Skip if we can't set the attribute
                    pass
        else:
            # Direct assignment for primitive types and other values
            try:
                setattr(target_obj, attr_name, source_value)
            except (AttributeError, TypeError):
                # Skip if we can't set the attribute (e.g., read-only property)
                pass


def deserialize(data, callback, _path=""):
    """
    Recursively deserialize a dict back into objects, handling primitives,
    containers, and custom class instances. Calls callback(obj, path) for every object
    and replaces obj with the callback's return value before continuing.

    Args:
        data: The serialized data to deserialize.
        callback: A function taking two arguments (the data dict and current path) and returning the (possibly transformed) object.
        _path: Internal parameter tracking the current path/member name.
    """
    # Run through callback first (object may be replaced)
    result = callback(data, _path)
    if result is not data:
        return result

    # If not a dict with __type__, return as-is
    if not isinstance(data, dict) or "__type__" not in data:
        return data

    type_name = data["__type__"]

    # Primitive types
    if type_name in ("str", "int", "float", "bool", "NoneType"):
        return data["value"]

    # NumPy scalar types
    if type_name.startswith("numpy."):
        if type_name == "numpy.ndarray":
            return deserialize_ndarray(data)
        else:
            # NumPy scalar types
            numpy_type = getattr(np, type_name.split(".")[-1])
            return numpy_type(data["value"])

    # Mappings (like dict)
    if type_name == "dict":
        return {k: deserialize(v, callback, f"{_path}.{k}" if _path else k) for k, v in data["items"].items()}

    # Iterables (like list, tuple, set)
    if type_name in ("list", "tuple", "set"):
        items = [
            deserialize(item, callback, f"{_path}[{i}]" if _path else f"[{i}]") for i, item in enumerate(data["items"])
        ]
        if type_name == "tuple":
            return tuple(items)
        elif type_name == "set":
            return set(items)
        else:
            return items

    # Custom objects
    if "attributes" in data:
        # For now, return a simple dict representation
        # In a full implementation, you might want to reconstruct the actual class
        return {
            attr: deserialize(value, callback, f"{_path}.{attr}" if _path else attr)
            for attr, value in data["attributes"].items()
        }

    # Unknown type - return the data as-is
    return data


# returns a model and a state history
def deserialize_newton(data: dict):
    """
    Deserialize Newton simulation data using callback approach.

    Args:
        data: The serialized data containing model and states.

    Returns:
        The deserialized data structure.
    """

    def callback(x, path):
        if isinstance(x, dict) and x.get("__type__") == "warp.array":
            # print(f"deserialize warp.array at path: {path}")
            return wp.array(deserialize_ndarray(x["data"]))

        if isinstance(x, dict) and x.get("__type__") == "warp.HashGrid":
            # print(f"deserialize warp.HashGrid at path: {path}")
            # Return None or create empty HashGrid as appropriate
            return None

        if isinstance(x, dict) and x.get("__type__") == "warp.Mesh":
            # print(f"deserialize warp.Mesh at path: {path}")
            # Return None or create empty Mesh as appropriate
            return None

        if isinstance(x, dict) and x.get("__type__") == "callable":
            # print(f"deserialize callable at path: {path}")
            # Return None for callables as they can't be serialized/deserialized
            return None

        return x

    result = deserialize(data, callback)

    # Extract model and states from the deserialized data
    if isinstance(result, dict) and "model" in result and "states" in result:
        model_dict = result["model"]
        state_history = result["states"]
    else:
        # Handle case where result is directly the model and states tuple
        model_dict, state_history = result

    model = Model()
    transfer_to_model(model_dict, model, None)

    return model, state_history


class BasicRecorder:
    """A class to record and playback simulation body transforms."""

    def __init__(self):
        """
        Initializes the Recorder.
        """
        self.transforms_history: list[wp.array] = []
        self.point_clouds_history: list[list[wp.array]] = []

    def record(self, body_transforms: wp.array, point_clouds: list[wp.array] | None = None):
        """
        Records a snapshot of body transforms.

        Args:
            body_transforms (wp.array): A warp array representing the body transforms.
                This is typically retrieved from `state.body_q`.
            point_clouds (list[wp.array] | None): An optional list of warp arrays representing point clouds.
        """
        self.transforms_history.append(wp.clone(body_transforms))
        if point_clouds:
            self.point_clouds_history.append([wp.clone(pc) for pc in point_clouds if pc is not None and pc.size > 0])
        else:
            self.point_clouds_history.append([])

    def playback(self, frame_id: int) -> tuple[wp.array | None, list[wp.array] | None]:
        """
        Plays back a recorded frame by returning the stored body transforms and point cloud.

        Args:
            frame_id (int): The integer index of the frame to be played back.

        Returns:
            A tuple containing the body transforms and point clouds for the given
            frame, or (None, None) if the frame_id is out of bounds.
        """
        if not (0 <= frame_id < len(self.transforms_history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return None, None

        transforms = self.transforms_history[frame_id]
        point_clouds = self.point_clouds_history[frame_id] if frame_id < len(self.point_clouds_history) else None
        return transforms, point_clouds

    def save_to_file(self, file_path: str):
        """
        Saves the recorded transforms history to a file.

        Args:
            file_path (str): The full path to the file where the transforms will be saved.
        """
        history_np = {f"frame_{i}": t.numpy() for i, t in enumerate(self.transforms_history)}
        for i, pc_list in enumerate(self.point_clouds_history):
            history_np[f"frame_{i}_points_count"] = len(pc_list)
            for j, pc in enumerate(pc_list):
                if pc is not None:
                    history_np[f"frame_{i}_points_{j}"] = pc.numpy()
        np.savez_compressed(file_path, **history_np)

    def load_from_file(self, file_path: str, device=None):
        """
        Loads recorded transforms from a file, replacing the current history.

        Args:
            file_path (str): The full path to the file from which to load the transforms.
            device: The device to load the transforms onto. If None, uses CPU.
        """
        self.transforms_history.clear()
        self.point_clouds_history.clear()
        with np.load(file_path) as data:
            try:
                transform_keys = [k for k in data.keys() if k.startswith("frame_") and "_points" not in k]
                frame_keys = sorted(transform_keys, key=lambda x: int(x.split("_")[1]))
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid frame key format in file: {e}") from e
            for key in frame_keys:
                frame_index_str = key.split("_")[1]
                transform_np = data[key]
                transform_wp = wp.array(transform_np, dtype=wp.transform, device=device)
                self.transforms_history.append(transform_wp)

                pc_list = []
                count_key = f"frame_{frame_index_str}_points_count"
                if count_key in data:
                    count = int(data[count_key].item())
                    for j in range(count):
                        points_key = f"frame_{frame_index_str}_points_{j}"
                        if points_key in data:
                            points_np = data[points_key]
                            points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
                            pc_list.append(points_wp)
                self.point_clouds_history.append(pc_list)


class ModelAndStateRecorder:
    """A class to record and playback simulation model and state using pickle serialization.

    WARNING: This class uses pickle for serialization which is UNSAFE and can execute
    arbitrary code when loading files. Only load recordings from TRUSTED sources that
    you have verified. Loading recordings from untrusted sources could lead to malicious
    code execution and compromise your system."""

    def __init__(self):
        """
        Initializes the Recorder.
        """
        self.history: list[dict] = []
        self.model_data: dict = {}

    def _get_device_from_state(self, state: State):
        """
        Retrieves the device from a simulation state object.

        This is done by finding the first `wp.array` attribute in the state
        and returning its device.

        Args:
            state (State): The simulation state.

        Returns:
            The device of the state's arrays, or None if no wp.array is found.
        """
        # device can be retrieved from any warp array attribute in the state
        for _name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                return value.device
        return None

    def record(self, state: State):
        """
        Records a snapshot of the state.

        Args:
            state (State): The simulation state.
        """
        state_data = {}
        for name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                state_data[name] = value.numpy()
        self.history.append(state_data)

    def playback(self, state: State, frame_id: int):
        """
        Plays back a recorded frame by updating the state.

        Args:
            state (State): The simulation state to restore.
            frame_id (int): The integer index of the frame to be played back.
        """
        if not (0 <= frame_id < len(self.history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return

        state_data = self.history[frame_id]
        try:
            device = self._get_device_from_state(state)
        except ValueError:
            print("Warning: Unable to determine device from state. Playback skipped.")
            return

        for name, value_np in state_data.items():
            if hasattr(state, name):
                value_wp = wp.array(value_np, device=device)
                setattr(state, name, value_wp)

    def record_model(self, model: Model):
        """
        Records a snapshot of the model's serializable attributes.
        It stores warp arrays as numpy arrays, and primitive types as-is.

        Args:
            model (Model): The simulation model.
        """
        self.raw_model = model
        model_data = {}
        for name, value in model.__dict__.items():
            if name == "shape_source":
                print("shape_source")
                print(value)

            if isinstance(value, wp.array):
                model_data[name] = value.numpy()
                if name == "shape_source2":
                    print("shape_source2")
                    print(value.numpy())
        self.model_data = model_data

    def playback_model(self, model: Model):
        """
        Plays back a recorded model by updating its attributes.

        Args:
            model (Model): The simulation model to restore.
        """
        if not self.model_data:
            print("Warning: No model data to playback.")
            return

        try:
            device = self._get_device_from_state(model)
        except ValueError:
            print("Warning: Unable to determine device from state. Playback skipped.")
            return

        for name, value_np in self.model_data.items():
            if hasattr(model, name):
                value_wp = wp.array(value_np, device=device)
                setattr(model, name, value_wp)

    def save_to_file(self, file_path: str):
        """
        Saves the recorded history to a file using pickle with unpicklable object handling.

        Args:
            file_path (str): The full path to the file.
        """
        data_to_save = {"model": self.raw_model, "states": self.history}
        serialized_data = serialize_newton(data_to_save)

        deserialized_model, deserialized_history = deserialize_newton(serialized_data)
        print("deserialization done")

        data_to_save2 = {"model": deserialized_model, "states": deserialized_history}
        serialized_data2 = serialize_newton(data_to_save2)

        # Save in a human readable format using JSON
        import json

        with open(file_path + ".json", "w") as f:
            json.dump(serialized_data, f, indent=4)

        with open(file_path + ".json2", "w") as f:
            json.dump(serialized_data2, f, indent=4)

        with open(file_path, "wb") as f:
            data_to_save = {"model": self.model_data, "states": self.history}
            pickle.dump(data_to_save, f)

    def load_from_file(self, file_path: str):
        """
        Loads a recorded history from a file, replacing the current history.

        Args:
            file_path (str): The full path to the file.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "states" in data:
                self.history = data.get("states", [])
                self.model_data = data.get("model", {})
            else:
                # For backward compatibility with old format.
                self.history = data
                self.model_data = {}
