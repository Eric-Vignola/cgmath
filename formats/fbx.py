import logging
import os
import pathlib

import fbx
import numpy as np

LOGGER = logging.getLogger(__name__)


"""
FBX Data manipulation class for Autodesk Maya
"""


class BaseData:
    """Base class for FBX data objects with common functionality"""

    def __init__(self, scene, data_object):
        """
        Initialize BaseData with scene and data object

        Args:
            scene (fbx.FbxScene): The FBX scene containing the data object
            data_object: The FBX data object (e.g., FbxAnimStack, FbxAnimLayer)
        """
        self._scene = scene
        self._data  = data_object

    @property
    def name(self):
        """
        Get the data object's name in Maya-friendly format

        Returns:
            str: The name of the data object converted to Maya naming conventions
        """
        fbx_name = self._data.GetName()
        return self._fbx_to_maya_name(fbx_name)

    def _fbx_to_maya_name(self, fbx_name):
        """
        Convert FBX naming convention back to Maya convention

        Args:
            fbx_name (str): FBX-style name

        Returns:
            str: Maya-style name
        """
        # FBX to Maya transform attribute translation
        fbx_to_maya_map = {
            "Lcl TranslationX": "translateX",
            "Lcl TranslationY": "translateY",
            "Lcl TranslationZ": "translateZ",
            "Lcl RotationX":    "rotateX",
            "Lcl RotationY":    "rotateY",
            "Lcl RotationZ":    "rotateZ",
            "Lcl ScalingX":     "scaleX",
            "Lcl ScalingY":     "scaleY",
            "Lcl ScalingZ":     "scaleZ",
            "Visibility":       "visibility",
        }

        # Handle blendshape naming: "mesh_blendshape.smile.DeformPercent" -> "mesh_blendshape.smile"
        if fbx_name.endswith(".DeformPercent"):
            return fbx_name[: -len(".DeformPercent")]

        # Handle transform attributes: "pCube1.Lcl TranslationX" -> "pCube1.translateX"
        for fbx_attr, maya_attr in fbx_to_maya_map.items():
            if fbx_name.endswith(f".{fbx_attr}"):
                node_part = fbx_name[: -len(f".{fbx_attr}")]
                return f"{node_part}.{maya_attr}"

        # If no conversion needed, return as-is
        return fbx_name

    def __contains__(self, item):
        """
        Check if item is contained in the object's name

        This enables operations like: 'word' in object_instance

        Args:
            item (str): The string to search for in the object's name

        Returns:
            bool: True if item is found in the object's name, False otherwise
        """
        return str(item) in self.name

    def __str__(self):
        """
        String representation of data object when cast to string

        Returns:
            str: The object's name without class wrapper
        """
        return self.name

    def __repr__(self):
        """
        String representation of data object

        Returns:
            str: String in format "ClassName('name_of_object')"
        """
        return f"{type(self).__name__}('{self.name}')"


class TakeData(BaseData):
    """Class to manipulate a specific take (animation stack) in an FBX scene"""

    def __init__(self, scene, anim_stack):
        """
        Initialize TakeData with scene and animation stack

        Args:
            scene (fbx.FbxScene): The FBX scene containing the take
            anim_stack (fbx.FbxAnimStack): The FBX animation stack object
        """
        super().__init__(scene, anim_stack)

    @property
    def layers(self):
        """
        Get all animation layers from this take

        Returns:
            LayerList: LayerList containing LayerData objects
        """
        layer_list = LayerList()
        layer_count = self._data.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        )

        for i in range(layer_count):
            anim_layer = self._data.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), i
            )
            layer_data = LayerData(self._scene, anim_layer)
            layer_list.append(layer_data)

        return layer_list

    def rename(self, name: str):
        """
        Rename this take to a new name

        Args:
            name (str): The new name for the take

        Raises:
            ValueError: If the new name already exists among other takes in the scene
            RuntimeError: If no scene is available for validation
        """
        if self._scene is None:
            raise RuntimeError("No scene available for take validation")

        # Convert to string to ensure we're working with strings
        new_name = str(name).strip()

        if not new_name:
            raise ValueError("Take name cannot be empty")

        # Get current name to avoid false positive when checking uniqueness
        current_name = self.name

        # Check if the new name already exists among other takes (excluding current take)
        stack_count = self._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        )

        existing_names = []
        for i in range(stack_count):
            anim_stack = self._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i
            )
            stack_name = anim_stack.GetName()
            if stack_name != current_name:  # Exclude current take
                existing_names.append(stack_name)

        if new_name in existing_names:
            raise ValueError(
                f"Take name '{new_name}' already exists. "
                f"Existing takes: {existing_names + [current_name]}"
            )

        # Set the new name on the animation stack
        self._data.SetName(new_name)

    def create_layer(self, name: str):
        """
        Create a new animation layer under this take

        Args:
            name (str): The name for the new layer

        Returns:
            LayerData: New LayerData object for the created layer

        Raises:
            ValueError: If the layer name already exists in this take or if name is empty
            RuntimeError: If no scene is available for layer creation
        """
        if self._scene is None:
            raise RuntimeError("No scene available for layer creation")

        # Convert to string to ensure we're working with strings
        new_name = str(name).strip()

        if not new_name:
            raise ValueError("Layer name cannot be empty")

        # Check if the new name already exists among layers in this take
        layer_count = self._data.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        )

        existing_names = []
        for i in range(layer_count):
            layer = self._data.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), i
            )
            layer_name = layer.GetName()
            existing_names.append(layer_name)

        if new_name in existing_names:
            raise ValueError(
                f"Layer name '{new_name}' already exists in take '{self.name}'. "
                f"Existing layers: {existing_names}"
            )

        # Create the new animation layer
        new_layer = fbx.FbxAnimLayer.Create(self._scene, new_name)

        # Add the layer to this animation stack (take)
        self._data.AddMember(new_layer)

        # Return a LayerData object
        return LayerData(self._scene, new_layer)

    @property
    def start_time(self):
        """
        Get the start time of this take

        Returns:
            float: The start time in seconds, or None if no scene is available
        """
        if self._scene is None:
            return None

        # Get the local time span from the animation stack
        local_time_span = self._data.GetLocalTimeSpan()
        start_time      = local_time_span.GetStart()

        # Convert FbxTime to seconds (same units as CurveData.times)
        return start_time.GetSecondDouble()

    @property
    def end_time(self):
        """
        Get the end time of this take

        Returns:
            float: The end time in seconds, or None if no scene is available
        """
        if self._scene is None:
            return None

        # Get the local time span from the animation stack
        local_time_span = self._data.GetLocalTimeSpan()
        end_time        = local_time_span.GetStop()

        # Convert FbxTime to seconds (same units as CurveData.times)
        return end_time.GetSecondDouble()

    @start_time.setter
    def start_time(self, time_value):
        """
        Set the start time of this take

        Args:
            time_value (float): The start time in seconds

        Raises:
            RuntimeError: If no scene is available
            ValueError: If time value is not a finite number
        """
        if self._scene is None:
            raise RuntimeError("No scene available to set start time")

        # Validate input
        try:
            time_float = float(time_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Time value must be a number, got {type(time_value).__name__}"
            )

        if not np.isfinite(time_float):
            raise ValueError("Time value must be a finite number")

        # Get current time span
        local_time_span  = self._data.GetLocalTimeSpan()
        current_end_time = local_time_span.GetStop()

        # Create new start time
        new_start_time = fbx.FbxTime()
        new_start_time.SetSecondDouble(time_float)

        # Create new time span with updated start time
        new_time_span = fbx.FbxTimeSpan(new_start_time, current_end_time)
        self._data.SetLocalTimeSpan(new_time_span)

    @end_time.setter
    def end_time(self, time_value):
        """
        Set the end time of this take

        Args:
            time_value (float): The end time in seconds

        Raises:
            RuntimeError: If no scene is available
            ValueError: If time value is not a finite number
        """
        if self._scene is None:
            raise RuntimeError("No scene available to set end time")

        # Validate input
        try:
            time_float = float(time_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Time value must be a number, got {type(time_value).__name__}"
            )

        if not np.isfinite(time_float):
            raise ValueError("Time value must be a finite number")

        # Get current time span
        local_time_span    = self._data.GetLocalTimeSpan()
        current_start_time = local_time_span.GetStart()

        # Create new end time
        new_end_time = fbx.FbxTime()
        new_end_time.SetSecondDouble(time_float)

        # Create new time span with updated end time
        new_time_span = fbx.FbxTimeSpan(current_start_time, new_end_time)
        self._data.SetLocalTimeSpan(new_time_span)

    @property
    def start_frame(self):
        """
        Get the start time of this take expressed in frames

        Returns:
            float: The start time in frames, or None if no scene is available or fps cannot be determined
        """
        if self._scene is None:
            return None

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            return None

        # Get start time in seconds and convert to frames
        start_time_seconds = self.start_time
        if start_time_seconds is None:
            return None

        return start_time_seconds * fps

    @start_frame.setter
    def start_frame(self, frame_value):
        """
        Set the start time of this take using frame values

        Args:
            frame_value (float): The start time in frames

        Raises:
            RuntimeError: If no scene is available or fps cannot be determined
            ValueError: If frame value is not a finite number
        """
        if self._scene is None:
            raise RuntimeError("No scene available to convert frames to seconds")

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            raise RuntimeError("Scene fps is not available for frame conversion")

        # Validate input
        try:
            frame_float = float(frame_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Frame value must be a number, got {type(frame_value).__name__}"
            )

        if not np.isfinite(frame_float):
            raise ValueError("Frame value must be a finite number")

        # Convert frames to seconds
        time_seconds = frame_float / fps

        # Use the existing start_time setter
        self.start_time = time_seconds

    @property
    def end_frame(self):
        """
        Get the end time of this take expressed in frames

        Returns:
            float: The end time in frames, or None if no scene is available or fps cannot be determined
        """
        if self._scene is None:
            return None

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            return None

        # Get end time in seconds and convert to frames
        end_time_seconds = self.end_time
        if end_time_seconds is None:
            return None

        return end_time_seconds * fps

    @end_frame.setter
    def end_frame(self, frame_value):
        """
        Set the end time of this take using frame values

        Args:
            frame_value (float): The end time in frames

        Raises:
            RuntimeError: If no scene is available or fps cannot be determined
            ValueError: If frame value is not a finite number
        """
        if self._scene is None:
            raise RuntimeError("No scene available to convert frames to seconds")

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            raise RuntimeError("Scene fps is not available for frame conversion")

        # Validate input
        try:
            frame_float = float(frame_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Frame value must be a number, got {type(frame_value).__name__}"
            )

        if not np.isfinite(frame_float):
            raise ValueError("Frame value must be a finite number")

        # Convert frames to seconds
        time_seconds = frame_float / fps

        # Use the existing end_time setter
        self.end_time = time_seconds

    @property
    def duration(self):
        """
        Get the duration of this take in seconds

        Returns:
            float: The duration in seconds (end_time - start_time), or None if no scene is available
        """
        if self._scene is None:
            return None

        start = self.start_time
        end   = self.end_time

        if start is None or end is None:
            return None

        return end - start

    @property
    def frame_count(self):
        """
        Get the duration of this take in number of frames

        Returns:
            float: The duration in frames, or None if no scene is available or fps cannot be determined
        """
        if self._scene is None:
            return None

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            return None

        # Get duration in seconds and convert to frames
        duration_seconds = self.duration
        if duration_seconds is None:
            return None

        return duration_seconds * fps

    @property
    def curves(self):
        """
        Get all animation curves from this take

        This is a convenience property that only works if the take has exactly one layer.
        If multiple layers exist, raises an error.

        Returns:
            CurveList: CurveList containing CurveData objects from the single layer

        Raises:
            RuntimeError: If the take has multiple layers
        """
        layers = self.layers

        if len(layers) == 0:
            # No layers, return empty CurveList
            return CurveList()
        elif len(layers) == 1:
            # Only one layer, return its curves
            return layers[0].curves
        else:
            # Multiple layers, raise error
            raise RuntimeError(
                f"Cannot access curves property on take '{self.name}' with multiple layers ({len(layers)} layers found). "
                f"Use take.layers[index].curves instead."
            )

    def create_curve(self, node_name, property_name):
        """
        Create a new animation curve for this take

        This is a convenience method that only works if the take has exactly one layer.
        If multiple layers exist, raises an error.

        Args:
            node_name (str): Name of the node containing the attribute
            property_name (str): Property name (Maya or FBX style)

        Returns:
            CurveData: New CurveData object for the created curve

        Raises:
            RuntimeError: If the take has multiple layers
            ValueError: If node or property not found
            RuntimeError: If curve creation fails or property is already animated
        """
        layers = self.layers

        if len(layers) == 0:
            # No layers - create a default "BaseLayer" first
            self.create_layer("BaseLayer")
            layers = self.layers

        if len(layers) == 1:
            # Only one layer, delegate to its create_curve method
            return layers[0].create_curve(node_name, property_name)
        else:
            # Multiple layers, raise error
            raise RuntimeError(
                f"Cannot create curve on take '{self.name}' with multiple layers ({len(layers)} layers found). "
                f"Use take.layers[index].create_curve(node_name, property_name) instead."
            )


class LayerData(BaseData):
    """Class to manipulate a specific animation layer in an FBX scene"""

    def __init__(self, scene, anim_layer):
        """
        Initialize LayerData with scene and animation layer

        Args:
            scene (fbx.FbxScene): The FBX scene containing the layer
            anim_layer (fbx.FbxAnimLayer): The FBX animation layer object
        """
        super().__init__(scene, anim_layer)

    @property
    def curves(self):
        """
        Get all animation curves in this layer

        Returns:
            CurveList: CurveList containing CurveData objects
        """
        curve_list = CurveList()

        # Get all animation curve nodes from this layer
        curve_node_count = self._data.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
        )

        for i in range(curve_node_count):
            curve_node = self._data.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId), i
            )

            # Method 1: Check properties that this curve node is connected to
            property_count = curve_node.GetDstPropertyCount()
            for j in range(property_count):
                prop = curve_node.GetDstProperty(j)
                if prop.IsValid():
                    prop_object = prop.GetFbxObject()

                    # Handle different types of objects (nodes, geometries, etc.)
                    node_name = None
                    if isinstance(prop_object, fbx.FbxNode):
                        node_name = prop_object.GetName()
                    elif hasattr(prop_object, "GetNode"):
                        # For things like FbxGeometry, FbxNodeAttribute, etc.
                        try:
                            node = prop_object.GetNode(0)  # Get first node
                            if node:
                                node_name = node.GetName()
                        except BaseException:
                            # If GetNode(0) fails, try without parameters
                            try:
                                node = prop_object.GetNode()
                                if node:
                                    node_name = node.GetName()
                            except:
                                pass
                    else:
                        # Try to get the name directly for other object types
                        try:
                            node_name = prop_object.GetName()
                        except:
                            continue

                    if node_name:
                        attribute_name = prop.GetName()

                        # Get channels count from the curve node
                        channels_count = curve_node.GetChannelsCount()

                        if channels_count == 1:
                            # Single channel (like visibility, blendShape weights, custom attributes)
                            anim_curve = curve_node.GetCurve(0)
                            if anim_curve:
                                curve_key = f"{node_name}.{attribute_name}"
                                curve_data = CurveData(
                                    self._scene, anim_curve, curve_key
                                )
                                curve_list.append(curve_data)
                        else:
                            # Multiple channels (like translateX, translateY, translateZ)
                            for k in range(channels_count):
                                anim_curve = curve_node.GetCurve(k)
                                if anim_curve:
                                    # Get channel name (X, Y, Z, etc.)
                                    channel_name = curve_node.GetChannelName(k)
                                    if channel_name:
                                        curve_key = f"{node_name}.{attribute_name}{channel_name}"
                                    else:
                                        curve_key = f"{node_name}.{attribute_name}[{k}]"
                                    curve_data = CurveData(
                                        self._scene, anim_curve, curve_key
                                    )
                                    curve_list.append(curve_data)

        return curve_list

    def rename(self, name: str):
        """
        Rename this layer to a new name

        Args:
            name (str): The new name for the layer

        Raises:
            ValueError: If the new name already exists among other layers in the same take
            RuntimeError: If no scene is available for validation or parent take cannot be found
        """
        if self._scene is None:
            raise RuntimeError("No scene available for layer validation")

        # Convert to string to ensure we're working with strings
        new_name = str(name).strip()

        if not new_name:
            raise ValueError("Layer name cannot be empty")

        # Get current name to avoid false positive when checking uniqueness
        current_name = self.name

        # Find the parent animation stack (take) that contains this layer
        parent_stack = None
        stack_count = self._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        )

        for i in range(stack_count):
            anim_stack = self._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i
            )

            # Check if this stack contains our layer
            layer_count = anim_stack.GetSrcObjectCount(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
            )

            for j in range(layer_count):
                layer = anim_stack.GetSrcObject(
                    fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), j
                )
                if layer == self._data:  # Found our layer
                    parent_stack = anim_stack
                    break

            if parent_stack:
                break

        if parent_stack is None:
            raise RuntimeError("Could not find parent animation stack for this layer")

        # Check if the new name already exists among other layers in the same take (excluding current layer)
        layer_count = parent_stack.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        )

        existing_names = []
        for i in range(layer_count):
            layer = parent_stack.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), i
            )
            layer_name = layer.GetName()
            if layer != self._data:  # Exclude current layer
                existing_names.append(layer_name)

        if new_name in existing_names:
            raise ValueError(
                f"Layer name '{new_name}' already exists in take '{parent_stack.GetName()}'. "
                f"Existing layers: {existing_names + [current_name]}"
            )

        # Set the new name on the animation layer
        self._data.SetName(new_name)

    def create_curve(self, node_name, property_name):
        """
        Create a new animation curve for an unkeyed attribute

        Supports both Maya and FBX naming conventions:
        - Maya transforms: "translateX" -> "Lcl TranslationX"
        - Maya blendshapes: node="face_mesh", property="smile" -> "face_mesh.smile.DeformPercent"
        - FBX naming: Direct usage of FBX property names

        Args:
            node_name (str): Name of the node containing the attribute
            property_name (str): Property name (Maya or FBX style)

        Returns:
            CurveData: New CurveData object for the created curve

        Raises:
            ValueError: If node or property not found
            RuntimeError: If curve creation fails or property is already animated
        """
        if self._scene is None:
            raise RuntimeError("No scene available for curve creation")

        # Store original names for curve key generation
        original_node_name     = node_name
        original_property_name = property_name

        # Maya-to-FBX attribute name translation
        translated_node_name, translated_property_name = self._translate_maya_to_fbx(
            node_name, property_name
        )

        # Use translated names for FBX operations
        node_name     = translated_node_name
        property_name = translated_property_name

        # Find the target node
        target_node = None
        root_node   = self._scene.GetRootNode()

        def find_node_recursive(node, name):
            if node.GetName() == name:
                return node
            for i in range(node.GetChildCount()):
                found = find_node_recursive(node.GetChild(i), name)
                if found:
                    return found
            return None

        target_node = find_node_recursive(root_node, node_name)

        # If not found as regular node, search for blendshape channel
        if target_node is None:
            # Search through blendshape channels
            blendshape_channel_count = self._scene.GetSrcObjectCount(
                fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId)
            )

            for i in range(blendshape_channel_count):
                channel = self._scene.GetSrcObject(
                    fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId), i
                )
                channel_name = channel.GetName()

                # Check if this matches our node name
                if channel_name == node_name:
                    target_node = channel
                    break

                # Also check for constructed blendshape.channel format
                # Find parent blendshape to construct full name
                blendshape_count = self._scene.GetSrcObjectCount(
                    fbx.FbxCriteria.ObjectType(fbx.FbxBlendShape.ClassId)
                )

                for j in range(blendshape_count):
                    blendshape = self._scene.GetSrcObject(
                        fbx.FbxCriteria.ObjectType(fbx.FbxBlendShape.ClassId), j
                    )

                    # Check if this blendshape contains the channel
                    channel_count = blendshape.GetBlendShapeChannelCount()
                    for k in range(channel_count):
                        if blendshape.GetBlendShapeChannel(k) == channel:
                            blendshape_name = blendshape.GetName()

                            # Construct full name and check for match
                            if channel_name.startswith(f"{blendshape_name}."):
                                full_channel_name = channel_name
                            else:
                                full_channel_name = f"{blendshape_name}.{channel_name}"

                            if full_channel_name == node_name:
                                target_node = channel
                                break

                    if target_node:
                        break

                if target_node:
                    break

        if target_node is None:
            raise ValueError(f"Node '{node_name}' not found in scene")

        # Parse property name to handle channeled properties and get channel name
        base_property_name = property_name
        channel_name       = None

        # Handle common channeled properties
        channel_map = {
            "Lcl TranslationX": ("Lcl Translation", "X"),
            "Lcl TranslationY": ("Lcl Translation", "Y"),
            "Lcl TranslationZ": ("Lcl Translation", "Z"),
            "Lcl RotationX":    ("Lcl Rotation", "X"),
            "Lcl RotationY":    ("Lcl Rotation", "Y"),
            "Lcl RotationZ":    ("Lcl Rotation", "Z"),
            "Lcl ScalingX":     ("Lcl Scaling", "X"),
            "Lcl ScalingY":     ("Lcl Scaling", "Y"),
            "Lcl ScalingZ":     ("Lcl Scaling", "Z"),
        }

        if property_name in channel_map:
            base_property_name, channel_name = channel_map[property_name]

        # Find the target property (using base property name)
        target_property = target_node.FindProperty(base_property_name)
        if not target_property.IsValid():
            raise ValueError(
                f"Property '{base_property_name}' not found on node '{node_name}'"
            )

        # Use the high-level GetCurve method which automatically creates curve nodes and connections
        # This is the key insight from ChatGPT's working solution
        if channel_name:
            # Multi-channel property (like Translation, Rotation, Scaling)
            anim_curve = target_property.GetCurve(self._data, channel_name, True)
        else:
            # Single-channel property (like Visibility, DeformPercent)
            anim_curve = target_property.GetCurve(self._data, True)

        if anim_curve is None:
            raise RuntimeError(
                f"Failed to create curve for '{node_name}.{property_name}'"
            )

        # Create the curve key for the CurveData object (using original names)
        curve_key = f"{original_node_name}.{original_property_name}"

        # Return a CurveData object
        return CurveData(self._scene, anim_curve, curve_key)

    def _translate_maya_to_fbx(self, node_name, property_name):
        """
        Translate Maya attribute names to FBX equivalents

        Args:
            node_name (str): Maya node name
            property_name (str): Maya property name

        Returns:
            tuple: (translated_node_name, translated_property_name)
        """
        # Maya transform attribute translation
        maya_transform_map = {
            "translateX": "Lcl TranslationX",
            "translateY": "Lcl TranslationY",
            "translateZ": "Lcl TranslationZ",
            "rotateX":    "Lcl RotationX",
            "rotateY":    "Lcl RotationY",
            "rotateZ":    "Lcl RotationZ",
            "scaleX":     "Lcl ScalingX",
            "scaleY":     "Lcl ScalingY",
            "scaleZ":     "Lcl ScalingZ",
            "visibility": "Visibility",
        }

        # First check if the combination could be a blendshape channel
        # before checking transform attributes (blendshapes can have transform-like names)
        potential_blendshape_channel = f"{node_name}.{property_name}"

        # Search through blendshape channels first
        blendshape_channel_count = self._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId)
        )

        for i in range(blendshape_channel_count):
            channel = self._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId), i
            )
            channel_name = channel.GetName()

            # Check if the channel name matches our potential blendshape channel
            if channel_name == potential_blendshape_channel:
                # Found a matching blendshape channel, check if it has DeformPercent property
                deform_property = channel.FindProperty("DeformPercent")
                if deform_property.IsValid():
                    # This is a blendshape channel: translate to FBX format
                    return potential_blendshape_channel, "DeformPercent"

        # Check for Maya transform attributes only after checking blendshapes
        if property_name in maya_transform_map:
            return node_name, maya_transform_map[property_name]

        # Check for known FBX patterns (already handled)
        fbx_patterns = [
            "Lcl Translation",
            "Lcl Rotation",
            "Lcl Scaling",
            "DeformPercent",
            "Visibility",
        ]

        # If it's already an FBX pattern, return as-is
        for pattern in fbx_patterns:
            if pattern in property_name:
                return node_name, property_name

        # Check for Maya blendshape pattern (node_name + property_name = blendshape node)
        # But be careful: avoid converting things that look like transforms
        potential_blendshape_node = f"{node_name}.{property_name}"

        # Don't treat as blendshape if it looks like a transform attribute
        if not property_name in maya_transform_map:
            # Check if this potential blendshape node exists in the scene
            root_node = self._scene.GetRootNode()

            def find_node_recursive(node, name):
                if node.GetName() == name:
                    return node
                for i in range(node.GetChildCount()):
                    found = find_node_recursive(node.GetChild(i), name)
                    if found:
                        return found
                return None

            # First check regular nodes
            blendshape_node = find_node_recursive(root_node, potential_blendshape_node)

            # If not found in regular nodes, check blendshape channels
            if blendshape_node is None:
                blendshape_channel_count = self._scene.GetSrcObjectCount(
                    fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId)
                )

                for i in range(blendshape_channel_count):
                    channel = self._scene.GetSrcObject(
                        fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId), i
                    )
                    if channel.GetName() == potential_blendshape_node:
                        blendshape_node = channel
                        break

            if blendshape_node is not None:
                # Check if it has DeformPercent property
                deform_property = blendshape_node.FindProperty("DeformPercent")
                if deform_property.IsValid():
                    # This is a Maya blendshape: translate to FBX format
                    return potential_blendshape_node, "DeformPercent"

        # No translation needed - return original
        return node_name, property_name


class CurveData(BaseData):
    """Class to manipulate a specific animation curve from FBX data"""

    def __init__(self, scene, anim_curve, curve_name):
        """
        Initialize CurveData with scene and animation curve

        Args:
            scene (fbx.FbxScene): The FBX scene containing the curve
            anim_curve (fbx.FbxAnimCurve): The FBX animation curve object
            curve_name (str): The name/key identifying this curve (e.g., "node.attribute")
        """
        super().__init__(scene, anim_curve)
        self._curve_name = curve_name

    @property
    def name(self):
        """
        Get the curve's identifier name in Maya-friendly format

        Returns:
            str: The name/key of the curve converted to Maya naming conventions
        """
        return self._fbx_to_maya_name(self._curve_name)

    @property
    def key_count(self):
        """
        Get the number of keyframes in the animation curve

        Returns:
            int: Number of keyframes in the curve
        """
        return self._data.KeyGetCount()

    @property
    def values(self):
        """
        Get all keyframe values from the animation curve

        For DeformPercent curves, converts from percentage (0-100) to float (0.0-1.0)

        Returns:
            numpy.ndarray: Array of float values from all keyframes in the curve
        """
        key_count    = self._data.KeyGetCount()
        values_array = np.zeros(key_count, dtype=np.float64)

        for i in range(key_count):
            values_array[i] = self._data.KeyGetValue(i)

        # Convert percentage to float for DeformPercent curves (100.0 -> 1.0)
        if self._is_deform_percent_curve():
            values_array = values_array / 100.0

        return values_array

    @values.setter
    def values(self, value_data):
        """
        Set keyframe values for the animation curve

        For DeformPercent curves, converts from float (0.0-1.0) to percentage (0-100)

        This method updates values for existing keyframes without clearing them.
        The number of values must match the current keyframe count.

        Args:
            value_data (list or numpy.ndarray): Array of values for existing keyframes

        Raises:
            ValueError: If input is not a list or numpy array, if values are not numeric,
                       or if array size doesn't match current keyframe count
            RuntimeError: If no keyframes exist in the curve
        """
        current_count = self.key_count

        if current_count == 0:
            raise RuntimeError(
                "No keyframes exist in the curve. Use 'times' setter to create keyframes first."
            )

        # Convert input to numpy array for consistent handling
        if isinstance(value_data, (list, tuple)):
            values_array = np.array(value_data, dtype=np.float64)
        elif isinstance(value_data, np.ndarray):
            values_array = value_data.astype(np.float64)
        else:
            raise ValueError(
                f"Expected list or numpy array, got {type(value_data).__name__}"
            )

        # Validate that all values are numeric
        if not np.all(np.isfinite(values_array)):
            raise ValueError("All values must be finite numbers")

        # Check that array size matches current keyframe count
        if len(values_array) != current_count:
            raise ValueError(
                f"Array size ({len(values_array)}) must match current keyframe count ({current_count})"
            )

        # Convert float to percentage for DeformPercent curves (0.75 -> 75.0)
        if self._is_deform_percent_curve():
            values_array = values_array * 100.0

        # Begin keyframe modification for performance
        self._data.KeyModifyBegin()

        try:
            # Update values for existing keyframes
            for i, value in enumerate(values_array):
                self._data.KeySetValue(i, float(value))

        finally:
            # Always end keyframe modification, even if an error occurs
            self._data.KeyModifyEnd()

    @property
    def times(self):
        """
        Get all keyframe times from the animation curve

        Returns:
            numpy.ndarray: Array of float values (in seconds) from all keyframes in the curve
        """
        key_count   = self._data.KeyGetCount()
        times_array = np.zeros(key_count, dtype=np.float64)

        for i in range(key_count):
            fbx_time = self._data.KeyGetTime(i)
            # Convert FbxTime to seconds
            times_array[i] = fbx_time.GetSecondDouble()

        return times_array

    @times.setter
    def times(self, time_values):
        """
        Set keyframe times for the animation curve

        Args:
            time_values (list or numpy.ndarray): Array of time values in seconds

        Raises:
            ValueError: If input is not a list or numpy array, or if values are not numeric
        """
        # Clear existing keyframes first
        self.clear()

        # Convert input to numpy array for consistent handling
        if isinstance(time_values, (list, tuple)):
            times_array = np.array(time_values, dtype=np.float64)
        elif isinstance(time_values, np.ndarray):
            times_array = time_values.astype(np.float64)
        else:
            raise ValueError(
                f"Expected list or numpy array, got {type(time_values).__name__}"
            )

        # Validate that all values are numeric
        if not np.all(np.isfinite(times_array)):
            raise ValueError("All time values must be finite numbers")

        # Begin keyframe modification for performance
        self._data.KeyModifyBegin()

        try:
            # Add keyframes with specified times and default value of 0.0
            for time_seconds in times_array:
                # Create FbxTime object
                fbx_time = fbx.FbxTime()
                fbx_time.SetSecondDouble(float(time_seconds))

                # Add keyframe and get its index
                add_result = self._data.KeyAdd(fbx_time)
                key_index = (
                    add_result[0] if isinstance(add_result, tuple) else add_result
                )

                # Set default value of 0.0 for new keyframes
                self._data.KeySetValue(key_index, 0.0)

        finally:
            # Always end keyframe modification, even if an error occurs
            self._data.KeyModifyEnd()

    @property
    def frames(self):
        """
        Get all keyframe times from the animation curve expressed in frames

        Returns:
            numpy.ndarray: Array of float values (in frames) from all keyframes in the curve,
                         or None if no scene is available to get fps
        """
        if self._scene is None:
            return None

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            return None

        # Get times in seconds and convert to frames
        times_seconds = self.times
        frames_array  = times_seconds * fps

        return frames_array

    @frames.setter
    def frames(self, frame_values):
        """
        Set keyframe times for the animation curve using frame values

        Args:
            frame_values (list or numpy.ndarray): Array of frame values

        Raises:
            ValueError: If input is not a list or numpy array, or if values are not numeric
            RuntimeError: If no scene is available or fps cannot be determined
        """
        if self._scene is None:
            raise RuntimeError("No scene available to convert frames to seconds")

        # Get the scene's fps
        scene_data        = SceneData()
        scene_data._scene = self._scene
        fps               = scene_data.fps

        if fps is None:
            raise RuntimeError("Scene fps is not available for frame conversion")

        # Convert input to numpy array for consistent handling
        if isinstance(frame_values, (list, tuple)):
            frames_array = np.array(frame_values, dtype=np.float64)
        elif isinstance(frame_values, np.ndarray):
            frames_array = frame_values.astype(np.float64)
        else:
            raise ValueError(
                f"Expected list or numpy array, got {type(frame_values).__name__}"
            )

        # Validate that all values are numeric
        if not np.all(np.isfinite(frames_array)):
            raise ValueError("All frame values must be finite numbers")

        # Convert frames to seconds
        times_array = frames_array / fps

        # Use the existing times setter to set the keyframes
        self.times = times_array

    def clear(self):
        """
        Remove all keyframes from the animation curve while preserving attribute connections

        This method removes keyframes individually from end to beginning to maintain
        the connection between the curve and its driven attribute, unlike KeyClear()
        which can break these connections.
        """
        # Begin keyframe modification for performance
        self._data.KeyModifyBegin()

        # Remove all existing keyframes individually (preserves connections)
        key_count = self._data.KeyGetCount()
        for i in range(key_count - 1, -1, -1):  # Remove from end to beginning
            self._data.KeyRemove(i)

        # End keyframe modification
        self._data.KeyModifyEnd()

    def _is_deform_percent_curve(self):
        """
        Check if this curve is a DeformPercent curve (blendshape animation)

        Returns:
            bool: True if this is a DeformPercent curve, False otherwise
        """
        # Check FBX-style naming first
        if self._curve_name.endswith("DeformPercent"):
            return True

        # For Maya-style curves, we need to check if this curve actually refers to a blendshape channel
        # by looking in the scene data, not just doing string pattern matching
        if "." in self._curve_name and self._scene is not None:
            # Check if this curve name matches a blendshape channel in the scene
            blendshape_channel_count = self._scene.GetSrcObjectCount(
                fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId)
            )

            for i in range(blendshape_channel_count):
                channel = self._scene.GetSrcObject(
                    fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId), i
                )
                channel_name = channel.GetName()

                # If the curve name matches a blendshape channel name, this is a blendshape curve
                if channel_name == self._curve_name:
                    return True

        return False

    def __repr__(self):
        """
        String representation using Maya convention

        Returns:
            str: String in format "CurveData('maya_style_name')"
        """
        maya_name = self._fbx_to_maya_name(self._curve_name)
        return f"CurveData('{maya_name}')"


class BaseList:
    """Base class for list containers with custom access methods"""

    # Class attribute to be overridden by subclasses to specify allowed object type
    _allowed_type = None

    def __init__(self, data_objects=None):
        """
        Initialize BaseList with a list of data objects

        Args:
            data_objects (list, optional): List of data objects

        Raises:
            TypeError: If any object in data_objects is not of the allowed type
        """
        self._data = []
        if data_objects:
            for obj in data_objects:
                self._validate_type(obj)
            self._data = list(data_objects)

    def _validate_type(self, obj):
        """
        Validate that the object is of the allowed type

        Args:
            obj: Object to validate

        Raises:
            TypeError: If object is not of the allowed type
        """
        if self._allowed_type is not None:
            if not isinstance(obj, self._allowed_type):
                raise TypeError(
                    f"{type(self).__name__} can only contain {self._allowed_type.__name__} objects, "
                    f"got {type(obj).__name__}"
                )

    def __repr__(self):
        """
        String representation of list

        Returns:
            str: String in format "ClassName([item_name,...])"
        """
        item_names = [item.name for item in self._data]
        return f"{type(self).__name__}({item_names})"

    def __iter__(self):
        """
        Make list iterable, yielding data objects

        Yields:
            object: Each data object in the list
        """
        for item in self._data:
            yield item

    def __getitem__(self, key):
        """
        Get data object by index or by name

        Args:
            key (int or str): Index (int) or item name (str)

        Returns:
            object: The data object at the specified index or with the specified name

        Raises:
            IndexError: If index is out of range
            KeyError: If item name is not found
        """
        if isinstance(key, int):
            return self._data[key]
        elif isinstance(key, str):
            for item in self._data:
                if item.name == key:
                    return item
            raise KeyError(f"Item '{key}' not found in {type(self).__name__}")
        else:
            raise TypeError("Key must be int (index) or str (item name)")

    def __len__(self):
        """
        Get the number of items in the list

        Returns:
            int: Number of data objects
        """
        return len(self._data)

    def __contains__(self, item):
        """
        Check if item is contained in any of the item names in the list

        This enables operations like: 'word' in list_instance

        Args:
            item (str): The string to search for in any of the item names

        Returns:
            bool: True if item is found in any item's name, False otherwise
        """
        search_str = str(item)
        for data_item in self._data:
            if search_str in data_item.name:
                return True
        return False

    def append(self, data_object):
        """
        Add a data object to the list

        Args:
            data_object: Data object to add

        Raises:
            TypeError: If data_object is not of the allowed type
        """
        self._validate_type(data_object)
        self._data.append(data_object)

    def delete(self, name: str):
        """
        Delete an object from the scene and remove it from this list

        Args:
            name (str): Name of the object to delete

        Raises:
            KeyError: If object name is not found
            RuntimeError: If object deletion fails
        """
        # Find the object by name
        object_to_delete = None
        object_index     = None

        for i, obj in enumerate(self._data):
            if obj.name == name:
                object_to_delete = obj
                object_index     = i
                break

        if object_to_delete is None:
            raise KeyError(f"Object '{name}' not found in {type(self).__name__}")

        # Perform subclass-specific deletion logic
        self._delete_object_from_scene(object_to_delete)

        # Remove the object from our list
        self._data.pop(object_index)

    def _delete_object_from_scene(self, obj):
        """
        Perform specific deletion logic for this object type.
        Must be implemented by subclasses.

        Args:
            obj: The object to delete from the scene

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _delete_object_from_scene method"
        )


class LayerList(BaseList):
    """Class to hold a list of LayerData objects with custom access methods"""

    # Restrict to LayerData objects only
    _allowed_type = LayerData

    def __init__(self, layer_data_objects=None):
        """
        Initialize LayerList with a list of LayerData objects

        Args:
            layer_data_objects (list, optional): List of LayerData objects
        """
        super().__init__(layer_data_objects)

    def _delete_object_from_scene(self, layer_to_delete):
        """
        Delete a layer from the scene, including all its curves

        Args:
            layer_to_delete (LayerData): The layer object to delete from the scene
        """
        # Get all curves from this layer and delete them first
        curves      = layer_to_delete.curves
        curve_names = [curve.name for curve in curves]  # Get names before deletion

        # Delete all curves in this layer
        for curve_name in curve_names:
            try:
                curves.delete(curve_name)
            except (KeyError, RuntimeError):
                # Continue if curve deletion fails - might already be deleted
                pass

        # Get the FBX animation layer object
        anim_layer = layer_to_delete._data

        # Find parent animation stack that contains this layer
        parent_stack = None
        if layer_to_delete._scene is not None:
            stack_count = layer_to_delete._scene.GetSrcObjectCount(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
            )

            for i in range(stack_count):
                stack = layer_to_delete._scene.GetSrcObject(
                    fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i
                )

                # Check if this stack contains our layer
                layer_count = stack.GetSrcObjectCount(
                    fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
                )

                for j in range(layer_count):
                    layer = stack.GetSrcObject(
                        fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), j
                    )
                    if layer == anim_layer:
                        parent_stack = stack
                        break

                if parent_stack:
                    break

        # Remove layer from parent stack if found
        if parent_stack:
            parent_stack.RemoveMember(anim_layer)

        # Destroy the animation layer object
        anim_layer.Destroy()


class CurveList(BaseList):
    """Class to hold a list of CurveData objects with custom access methods"""

    # Restrict to CurveData objects only
    _allowed_type = CurveData

    def __init__(self, curve_data_objects=None):
        """
        Initialize CurveList with a list of CurveData objects

        Args:
            curve_data_objects (list, optional): List of CurveData objects
        """
        super().__init__(curve_data_objects)

    def __repr__(self):
        """
        String representation using Maya convention

        Returns:
            str: String in format "CurveList(['curve1', 'curve2', ...])"
        """
        maya_names = [item._fbx_to_maya_name(item._curve_name) for item in self._data]
        return f"CurveList({maya_names})"

    def __getitem__(self, key):
        """
        Get data object by index or by name (supports both Maya and FBX naming)

        Args:
            key (int or str): Index (int) or item name (str in Maya or FBX format)

        Returns:
            CurveData: The curve data object at the specified index or with the specified name

        Raises:
            IndexError: If index is out of range
            KeyError: If item name is not found
        """
        if isinstance(key, int):
            return self._data[key]
        elif isinstance(key, str):
            for item in self._data:
                # Check both FBX format name and Maya format name
                fbx_name  = item.name  # This returns the FBX format
                maya_name = item._fbx_to_maya_name(item._curve_name)

                if fbx_name == key or maya_name == key:
                    return item
            raise KeyError(f"Item '{key}' not found in {type(self).__name__}")
        else:
            raise TypeError("Key must be int (index) or str (item name)")

    def delete(self, name: str):
        """
        Delete a curve from the scene and remove it from this list

        Args:
            name (str): Name of the curve to delete (supports both Maya and FBX naming)

        Raises:
            KeyError: If curve name is not found
            RuntimeError: If curve deletion fails
        """
        # Find the curve by name (override BaseList's name matching for CurveList)
        curve_to_delete = None
        curve_index     = None

        for i, curve in enumerate(self._data):
            fbx_name  = curve.name
            maya_name = curve._fbx_to_maya_name(curve._curve_name)

            if fbx_name == name or maya_name == name:
                curve_to_delete = curve
                curve_index     = i
                break

        if curve_to_delete is None:
            raise KeyError(f"Curve '{name}' not found in CurveList")

        # Perform subclass-specific deletion logic
        self._delete_object_from_scene(curve_to_delete)

        # Remove the curve from our list
        self._data.pop(curve_index)

    def _delete_object_from_scene(self, curve_to_delete):
        """
        Delete a curve from the scene

        Args:
            curve_to_delete (CurveData): The curve object to delete from the scene
        """
        # Get the FBX animation curve object
        anim_curve = curve_to_delete._data

        # Find and remove all connections to this curve
        # Animation curves are connected through animation curve nodes
        curve_node_count = curve_to_delete._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
        )

        for i in range(curve_node_count):
            curve_node = curve_to_delete._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId), i
            )

            # Check if this curve node contains our animation curve
            channels_count = curve_node.GetChannelsCount()
            for j in range(channels_count):
                if curve_node.GetCurve(j) == anim_curve:
                    # Remove the curve from this channel
                    curve_node.DisconnectFromChannel(j)
                    break

        # Destroy the animation curve object
        anim_curve.Destroy()


class TakeList(BaseList):
    """Class to hold a list of TakeData objects with custom access methods"""

    # Restrict to TakeData objects only
    _allowed_type = TakeData

    def __init__(self, take_data_objects=None):
        """
        Initialize TakeList with a list of TakeData objects

        Args:
            take_data_objects (list, optional): List of TakeData objects
        """
        super().__init__(take_data_objects)

    def _delete_object_from_scene(self, take_to_delete):
        """
        Delete a take from the scene, including all its layers and curves

        Args:
            take_to_delete (TakeData): The take object to delete from the scene
        """
        # Get all layers from this take and delete them (which will also delete their curves)
        layers      = take_to_delete.layers
        layer_names = [layer.name for layer in layers]  # Get names before deletion

        # Delete all layers in this take (this will also delete all curves in each layer)
        for layer_name in layer_names:
            try:
                layers.delete(layer_name)
            except (KeyError, RuntimeError):
                # Continue if layer deletion fails - might already be deleted
                pass

        # Get the FBX animation stack object
        anim_stack = take_to_delete._data

        # Remove the animation stack from the scene
        if take_to_delete._scene is not None:
            take_to_delete._scene.RemoveMember(anim_stack)

        # Destroy the animation stack object
        anim_stack.Destroy()


class SceneData(BaseData):
    """Class to manipulate animation data in FBX files from Autodesk Maya"""

    def __init__(self, filename=None):
        """
        Initialize SceneData with optional automatic loading

        Args:
            filename (str, optional): Path to FBX file to load automatically.
                                    If None, creates empty SceneData instance.
                                    Supports ~ username expansion.
        """
        # Initialize BaseData with None initially (scene will be set on load)
        super().__init__(None, None)
        self._manager  = None
        self._filename = None

        # If filename is provided, load it automatically
        if filename is not None:
            self.load(filename)

    @property
    def name(self):
        """
        Get the scene's name

        Returns:
            str: The name of the scene, or "No Scene" if no scene is loaded
        """
        if self._data is None:
            return "No Scene"
        return self._data.GetName()

    def rename(self, name: str):
        """
        Rename this scene to a new name

        Args:
            name (str): The new name for the scene

        Raises:
            ValueError: If the name is empty
            RuntimeError: If no scene is loaded
        """
        if self._scene is None:
            raise RuntimeError("No scene is loaded. Use load() method first.")

        # Convert to string to ensure we're working with strings
        new_name = str(name).strip()

        if not new_name:
            raise ValueError("Scene name cannot be empty")

        # Set the new name on the scene
        self._scene.SetName(new_name)

    def load(self, filename):
        """
        Load an FBX file and store its scene data

        Args:
            filename (str): Path to the FBX file to load (supports ~ username expansion)
        """
        # Expand username if present (e.g., "~/Desktop/file.fbx" -> "/Users/username/Desktop/file.fbx")
        expanded_filename = os.path.expanduser(filename)

        if not os.path.exists(expanded_filename):
            raise FileNotFoundError(f"FBX file not found: {expanded_filename}")

        # Create FBX manager if it doesn't exist
        if self._manager is None:
            self._manager = fbx.FbxManager.Create()
            ios           = fbx.FbxIOSettings.Create(self._manager, fbx.IOSROOT)
            self._manager.SetIOSettings(ios)

        # Create scene
        self._scene = fbx.FbxScene.Create(self._manager, "ImportedScene")

        # Create importer
        importer = fbx.FbxImporter.Create(self._manager, "Importer")

        # Initialize the importer
        if not importer.Initialize(
            expanded_filename, -1, self._manager.GetIOSettings()
        ):
            error = importer.GetStatus().GetErrorString()
            importer.Destroy()
            raise RuntimeError(f"Failed to initialize FBX importer: {error}")

        # Import the scene
        if not importer.Import(self._scene):
            error = importer.GetStatus().GetErrorString()
            importer.Destroy()
            raise RuntimeError(f"Failed to import FBX scene: {error}")

        # Clean up importer
        importer.Destroy()

        # Store filename for reference
        self._filename = filename

        # Set the scene as the data object for BaseData
        self._data = self._scene

        # Extract filename without path or extension and set as scene name
        base_filename = os.path.basename(expanded_filename)  # Remove path
        scene_name    = os.path.splitext(base_filename)[0]   # Remove extension
        self.rename(scene_name)

    @property
    def scene(self):
        """
        Get the FBX scene object

        Returns:
            fbx.FbxScene: The loaded FBX scene, or None if no file is loaded
        """
        return self._scene

    @property
    def fps(self):
        """
        Get the scene's frames per second value

        Returns:
            float: The scene's fps value, or None if no scene is loaded
        """
        if self._scene is None:
            return None

        time_mode = self._scene.GetGlobalSettings().GetTimeMode()
        fps_map = {
            fbx.FbxTime.EMode.eDefaultMode: 24.0,
            fbx.FbxTime.EMode.eFrames120: 120.0,
            fbx.FbxTime.EMode.eFrames100: 100.0,
            fbx.FbxTime.EMode.eFrames60: 60.0,
            fbx.FbxTime.EMode.eFrames50: 50.0,
            fbx.FbxTime.EMode.eFrames48: 48.0,
            fbx.FbxTime.EMode.eFrames30: 30.0,
            fbx.FbxTime.EMode.eFrames30Drop: 29.97,
            fbx.FbxTime.EMode.eFrames24: 24.0,
            fbx.FbxTime.EMode.eFrames1000: 1000.0,
            fbx.FbxTime.EMode.eFrames96: 96.0,
            fbx.FbxTime.EMode.eFrames72: 72.0,
            fbx.FbxTime.EMode.eFrames59dot94: 59.94,
        }
        return fps_map.get(time_mode, None)

    @fps.setter
    def fps(self, fps_value):
        """
        Set the scene's frames per second value

        Args:
            fps_value (str): FPS value as string (e.g., "24", "30", "29.97", "60")

        Raises:
            RuntimeError: If no scene is loaded
            ValueError: If the fps value is not supported
        """
        if self._scene is None:
            raise RuntimeError("No FBX scene loaded. Use load() method first.")

        # Convert string to float for comparison
        try:
            fps_float = float(fps_value)
        except ValueError:
            raise ValueError(
                f"Invalid fps value: '{fps_value}'. Must be a numeric string."
            )

        # Map fps values to FBX time modes
        fps_to_mode_map = {
            24.0:   fbx.FbxTime.EMode.eFrames24,
            29.97:  fbx.FbxTime.EMode.eFrames30Drop,
            30.0:   fbx.FbxTime.EMode.eFrames30,
            48.0:   fbx.FbxTime.EMode.eFrames48,
            50.0:   fbx.FbxTime.EMode.eFrames50,
            59.94:  fbx.FbxTime.EMode.eFrames59dot94,
            60.0:   fbx.FbxTime.EMode.eFrames60,
            72.0:   fbx.FbxTime.EMode.eFrames72,
            96.0:   fbx.FbxTime.EMode.eFrames96,
            100.0:  fbx.FbxTime.EMode.eFrames100,
            120.0:  fbx.FbxTime.EMode.eFrames120,
            1000.0: fbx.FbxTime.EMode.eFrames1000,
        }

        if fps_float not in fps_to_mode_map:
            supported_fps = list(fps_to_mode_map.keys())
            raise ValueError(
                f"FPS value '{fps_value}' is not supported. "
                f"Supported values: {supported_fps}"
            )

        # Set the time mode on the scene
        time_mode = fps_to_mode_map[fps_float]
        self._scene.GetGlobalSettings().SetTimeMode(time_mode)

    @property
    def linear_units(self):
        """
        Get the scene's linear units

        Returns:
            str: The scene's linear units (e.g., "meters", "centimeters"), or None if no scene is loaded
        """
        if self._scene is None:
            return None

        system_unit = self._scene.GetGlobalSettings().GetSystemUnit()

        if system_unit == fbx.FbxSystemUnit.mm:
            return "millimeters"
        elif system_unit == fbx.FbxSystemUnit.cm:
            return "centimeters"
        elif system_unit == fbx.FbxSystemUnit.dm:
            return "decimeters"
        elif system_unit == fbx.FbxSystemUnit.m:
            return "meters"
        elif system_unit == fbx.FbxSystemUnit.km:
            return "kilometers"
        elif system_unit == fbx.FbxSystemUnit.Inch:
            return "inches"
        elif system_unit == fbx.FbxSystemUnit.Foot:
            return "feet"
        elif system_unit == fbx.FbxSystemUnit.Mile:
            return "miles"
        elif system_unit == fbx.FbxSystemUnit.Yard:
            return "yards"

        return "unknown"

    @linear_units.setter
    def linear_units(self, units_value):
        """
        Set the scene's linear units

        Args:
            units_value (str): Linear units value (supports both short and long forms)
                             e.g., "m", "meters", "cm", "centimeters", "km", "kilometers"

        Raises:
            RuntimeError: If no scene is loaded
            ValueError: If the units value is not supported
        """
        if self._scene is None:
            raise RuntimeError("No FBX scene loaded. Use load() method first.")

        # Normalize input to lowercase for comparison
        units_lower = units_value.lower().strip()

        # Map units to FBX system units (support both short and long forms)
        units_map = {
            # Millimeters
            "mm":          fbx.FbxSystemUnit.mm,
            "millimeter":  fbx.FbxSystemUnit.mm,
            "millimeters": fbx.FbxSystemUnit.mm,
            # Centimeters
            "cm":          fbx.FbxSystemUnit.cm,
            "centimeter":  fbx.FbxSystemUnit.cm,
            "centimeters": fbx.FbxSystemUnit.cm,
            # Decimeters
            "dm":         fbx.FbxSystemUnit.dm,
            "decimeter":  fbx.FbxSystemUnit.dm,
            "decimeters": fbx.FbxSystemUnit.dm,
            # Meters
            "m":      fbx.FbxSystemUnit.m,
            "meter":  fbx.FbxSystemUnit.m,
            "meters": fbx.FbxSystemUnit.m,
            # Kilometers
            "km":         fbx.FbxSystemUnit.km,
            "kilometer":  fbx.FbxSystemUnit.km,
            "kilometers": fbx.FbxSystemUnit.km,
            # Inches
            "in":     fbx.FbxSystemUnit.Inch,
            "inch":   fbx.FbxSystemUnit.Inch,
            "inches": fbx.FbxSystemUnit.Inch,
            # Feet
            "ft":   fbx.FbxSystemUnit.Foot,
            "foot": fbx.FbxSystemUnit.Foot,
            "feet": fbx.FbxSystemUnit.Foot,
            # Miles
            "mi":    fbx.FbxSystemUnit.Mile,
            "mile":  fbx.FbxSystemUnit.Mile,
            "miles": fbx.FbxSystemUnit.Mile,
            # Yards
            "yd":    fbx.FbxSystemUnit.Yard,
            "yard":  fbx.FbxSystemUnit.Yard,
            "yards": fbx.FbxSystemUnit.Yard,
        }

        if units_lower not in units_map:
            supported_units = sorted(set(units_map.keys()))
            raise ValueError(
                f"Units value '{units_value}' is not supported. "
                f"Supported values: {supported_units}"
            )

        # Set the system unit on the scene
        system_unit = units_map[units_lower]
        self._scene.GetGlobalSettings().SetSystemUnit(system_unit)

    @property
    def scale_factor(self):
        """
        Get the scene's global scale factor

        Returns:
            float: The scene's global scale factor, or None if no scene is loaded
        """
        if self._scene is None:
            return None

        system_unit = self._scene.GetGlobalSettings().GetSystemUnit()
        return system_unit.GetScaleFactor()

    def create_take(self, name: str):
        """
        Create a new take (animation stack) in the scene with a default "BaseLayer"

        Args:
            name (str): The name for the new take

        Returns:
            TakeData: New TakeData object for the created take

        Raises:
            ValueError: If the take name already exists in the scene or if name is empty
            RuntimeError: If no scene is available for take creation
        """
        if self._scene is None:
            raise RuntimeError("No scene available for take creation")

        # Convert to string to ensure we're working with strings
        new_name = str(name).strip()

        if not new_name:
            raise ValueError("Take name cannot be empty")

        # Check if the new name already exists among takes in the scene
        stack_count = self._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        )

        existing_names = []
        for i in range(stack_count):
            anim_stack = self._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i
            )
            stack_name = anim_stack.GetName()
            existing_names.append(stack_name)

        if new_name in existing_names:
            raise ValueError(
                f"Take name '{new_name}' already exists. "
                f"Existing takes: {existing_names}"
            )

        # Create the new animation stack (take)
        new_stack = fbx.FbxAnimStack.Create(self._scene, new_name)

        # Add the stack to the scene
        self._scene.AddMember(new_stack)

        # Create a TakeData object for the new take
        take_data = TakeData(self._scene, new_stack)

        # Create a default "BaseLayer" in the new take
        take_data.create_layer("BaseLayer")

        # Return the TakeData object
        return take_data

    def save(self, filename=None, embed_media=True, file_format=-1):
        """
        Save the FBX scene to disk

        Args:
            filename (str, optional): Path to save the FBX file (supports ~ username expansion).
                                    If None, overwrites the original file.
            embed_media (bool): Whether to embed textures and media in the FBX file.
                              Defaults to True so that Maya can extract them when opening the file.
            file_format (int): File format index to use. Defaults to -1 (auto-detect format).
                             When -1 and embed_media=False, will automatically find ASCII format.
        """
        if self._scene is None:
            raise RuntimeError("No FBX scene loaded. Use load() method first.")

        # Use original filename if none provided
        if filename is None:
            if self._filename is None:
                raise ValueError("No filename provided and no original file loaded")
            filename = self._filename

        # Expand username if present (e.g., "~/Desktop/file.fbx" -> "/Users/username/Desktop/file.fbx")
        expanded_filename = os.path.expanduser(filename)

        # Ensure filename has .fbx extension
        if not expanded_filename.lower().endswith(".fbx"):
            expanded_filename += ".fbx"

        # Create exporter
        exporter = fbx.FbxExporter.Create(self._manager, "Exporter")

        # Determine file format (adapted from FbxCommon.py)
        if (
            file_format < 0
            or file_format >= self._manager.GetIOPluginRegistry().GetWriterFormatCount()
        ):
            file_format = self._manager.GetIOPluginRegistry().GetNativeWriterFormat()
            if not embed_media:
                # Look for ASCII format when not embedding media
                format_count = (
                    self._manager.GetIOPluginRegistry().GetWriterFormatCount()
                )
                for format_index in range(format_count):
                    if self._manager.GetIOPluginRegistry().WriterIsFBX(format_index):
                        desc = self._manager.GetIOPluginRegistry().GetWriterFormatDescription(
                            format_index
                        )
                        if "ascii" in desc:
                            file_format = format_index
                            break

        # Configure IO settings (adapted from FbxCommon.py)
        if not self._manager.GetIOSettings():
            ios = fbx.FbxIOSettings.Create(self._manager, fbx.IOSROOT)
            self._manager.SetIOSettings(ios)

        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, embed_media)
        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_SHAPE, True)
        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GOBO, True)
        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_ANIMATION, True)
        self._manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GLOBAL_SETTINGS, True)

        # Initialize the exporter
        result = exporter.Initialize(
            expanded_filename, file_format, self._manager.GetIOSettings()
        )
        if result == True:
            result = exporter.Export(self._scene)

        if not result:
            error = exporter.GetStatus().GetErrorString()
            exporter.Destroy()
            raise RuntimeError(f"Failed to export FBX scene: {error}")

        # Clean up exporter
        exporter.Destroy()

        # Return the expanded filename on successful save
        return expanded_filename

    @property
    def takes(self):
        """
        Get all takes (animation stacks) in the scene

        Returns:
            TakeList: TakeList containing TakeData objects
        """
        if self._scene is None:
            return TakeList()

        take_list = TakeList()
        stack_count = self._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        )

        for i in range(stack_count):
            anim_stack = self._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i
            )
            take_data = TakeData(self._scene, anim_stack)
            take_list.append(take_data)

        return take_list

    @property
    def nodes(self):
        """
        Get all nodes in the scene, including blendshape channels

        Returns:
            list: List of node names including blendshape channels in format "blendshape.channel"
        """
        if self._scene is None:
            return []

        nodes     = []
        root_node = self._scene.GetRootNode()

        # Collect regular nodes from the hierarchy
        def collect_nodes(node):
            nodes.append(node.GetName())
            for i in range(node.GetChildCount()):
                collect_nodes(node.GetChild(i))

        collect_nodes(root_node)

        # Collect blendshape channel objects
        blendshape_channel_count = self._scene.GetSrcObjectCount(
            fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId)
        )

        for i in range(blendshape_channel_count):
            channel = self._scene.GetSrcObject(
                fbx.FbxCriteria.ObjectType(fbx.FbxBlendShapeChannel.ClassId), i
            )
            channel_name = channel.GetName()

            # Find the parent blendshape to construct the full name
            parent_blendshape = None
            blendshape_count = self._scene.GetSrcObjectCount(
                fbx.FbxCriteria.ObjectType(fbx.FbxBlendShape.ClassId)
            )

            for j in range(blendshape_count):
                blendshape = self._scene.GetSrcObject(
                    fbx.FbxCriteria.ObjectType(fbx.FbxBlendShape.ClassId), j
                )

                # Check if this blendshape contains the channel
                channel_count = blendshape.GetBlendShapeChannelCount()
                for k in range(channel_count):
                    if blendshape.GetBlendShapeChannel(k) == channel:
                        parent_blendshape = blendshape
                        break

                if parent_blendshape:
                    break

            if parent_blendshape:
                # Construct the full blendshape channel name
                blendshape_name = parent_blendshape.GetName()

                # Check if channel name already starts with blendshape name to avoid duplication
                if channel_name.startswith(f"{blendshape_name}."):
                    # Channel name already includes blendshape prefix, use as-is
                    full_channel_name = channel_name
                else:
                    # Channel name doesn't include blendshape prefix, add it
                    full_channel_name = f"{blendshape_name}.{channel_name}"

                nodes.append(full_channel_name)

        return nodes[1:]

    def destroy(self):
        """Clean up FBX manager and scene"""
        if self._manager is not None:
            self._manager.Destroy()
            self._manager  = None
            self._scene    = None
            self._filename = None


class FbxExporter:
    """Export fbx files from pipeline components"""

    _ROTATE_ORDER_MAP = {
        0: fbx.EFbxRotationOrder.eEulerXYZ,
        1: fbx.EFbxRotationOrder.eEulerYZX,
        2: fbx.EFbxRotationOrder.eEulerZXY,
        3: fbx.EFbxRotationOrder.eEulerXZY,
        4: fbx.EFbxRotationOrder.eEulerYXZ,
        5: fbx.EFbxRotationOrder.eEulerZYX,
    }
    """dict: mapping HierarchyData rotation order to fbx rotation order"""

    def __init__(self):
        self._skeleton = None
        self._meshes   = set()

        self._manager  = None
        self._scene    = None

    @property
    def manager(self):
        """fbx.FbxManager: the fbx manager that handles the memory"""

        if self._manager is None:
            self._manager = fbx.FbxManager.Create()
            # import/export requires IO settings
            self._manager.SetIOSettings(
                fbx.FbxIOSettings.Create(self._manager, fbx.IOSROOT)
            )
        return self._manager

    @property
    def scene(self):
        """fbx.FbxScene: the fbx scene to export"""

        if self._scene is None:
            self._scene = fbx.FbxScene.Create(self.manager, "")

            # maya coordinate system
            self._scene.GetGlobalSettings().SetAxisSystem(fbx.FbxAxisSystem.MayaYUp)
        return self._scene

    def add_skeleton(self, skeleton_component) -> None:
        """Add a skeleton component to the scene.

        Notes:
            Only a single skeleton is supported.
        """
        if self._skeleton is not None:
            raise ValueError(f"Skeleton already set: {self._skeleton}!")
        self._skeleton = skeleton_component

        for item in self._skeleton:
            node = fbx.FbxNode.Create(self.manager, item.name)

            node.LclTranslation.Set(fbx.FbxDouble3(*item.translate))
            node.LclRotation.Set(fbx.FbxDouble3(*item.rotate))
            node.LclScaling.Set(fbx.FbxDouble3(*item.scale))

            node.SetRotationOrder(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                self._ROTATE_ORDER_MAP[item.rotate_order],
            )

            if item.parent_node is None:
                parent_node = self.scene.GetRootNode()
            else:
                parent_node = self.scene.FindNodeByName(item.get_parent().name)

            parent_node.AddChild(node)

            if item.node_type == "joint":
                node_attr = fbx.FbxSkeleton.Create(self.manager, "")
                node_attr.SetSkeletonType(
                    fbx.FbxSkeleton.EType.eRoot
                    if item.parent_node is None
                    else fbx.FbxSkeleton.EType.eLimbNode
                )
                inherit_type = (
                    fbx.FbxTransform.EInheritType.eInheritRrs
                    if item.segment_scale_compensate
                    else fbx.FbxTransform.EInheritType.eInheritRSrs
                )
                node.SetTransformationInheritType(inherit_type)
                # turn on rotationActive or pre/post are ignored
                node.SetRotationActive(True)
                node.SetPreRotation(
                    fbx.FbxNode.EPivotSet.eSourcePivot,
                    fbx.FbxVector4(*item.joint_orient, 1.0),
                )
                # fbx composes post rotation as its inverse
                post = fbx.FbxAMatrix()
                post.SetR(fbx.FbxVector4(*item.rotate_axis, 1.0))
                node.SetPostRotation(
                    fbx.FbxNode.EPivotSet.eSourcePivot,
                    post.Inverse().GetR(),
                )
            elif item.node_type == "transform" or item.node_type == "space_transform":
                node_attr = fbx.FbxNull.Create(self.manager, "")
            elif item.node_type == "locator":
                node_attr = fbx.FbxMarker.Create(self.manager, "")
            else:
                raise RuntimeError(
                    f"Not sure how to process nodes of type: {item.node_type}!"
                )

            # set user defined attrs
            ud_attrs = item.user_defined_attributes
            if ud_attrs:
                attr_types = {
                    "string": fbx.FbxStringDT,
                    "double": fbx.FbxDoubleDT,
                    "int":    fbx.FbxIntDT,
                    "bool":   fbx.FbxBoolDT,
                    "short":  fbx.FbxShortDT,
                }
                for name, deets in ud_attrs.items():
                    if "attributeType" in deets:
                        fbx_att = attr_types[deets["attributeType"]]
                    else:
                        fbx_att = attr_types[deets["dataType"]]

                    prop = fbx.FbxProperty.Create(node, fbx_att, name, name)
                    prop.Set(deets["value"])

            node.SetNodeAttribute(node_attr)
        LOGGER.info(f"added skeleton: {skeleton_component}")

    def export(self, path: pathlib.Path, as_ascii=False, zero_root=False) -> None:
        """Export the scene to the given path"""
        exporter    = fbx.FbxExporter.Create(self.manager, "")
        file_format = self.manager.GetIOPluginRegistry().GetNativeWriterFormat()
        if as_ascii:
            file_format = self._get_acsii_type()

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if not exporter.Initialize(
            path.as_posix(), file_format, self.manager.GetIOSettings()
        ):
            raise Exception(
                f"Failed to initialize exporter: {exporter.GetStatus().GetErrorString()}"
            )

        # ensure parent dir exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # zero-out root translations and rotations
        if zero_root:
            skel_root = self._skeleton.data.get_roots()[0]
            root_name = skel_root.name

            LOGGER.debug(f"Zero-ing Root: {root_name}")
            skel_root = self.scene.FindNodeByName(root_name)
            if not skel_root:
                raise RuntimeError(f"Could not find {root_name} to reset!")

            skel_root.LclTranslation.Set(fbx.FbxDouble3(0, 0, 0))
            skel_root.LclRotation.Set(fbx.FbxDouble3(0, 0, 0))
            skel_root.LclScaling.Set(fbx.FbxDouble3(1, 1, 1))

            skel_root.SetRotationActive(True)
            skel_root.SetPreRotation(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                fbx.FbxVector4(0, 0, 0, 1.0),
            )
            skel_root.SetPostRotation(
                fbx.FbxNode.EPivotSet.eSourcePivot,
                fbx.FbxVector4(0, 0, 0, 1.0),
            )

        if not exporter.Export(self.scene):
            exporter.Destroy()
            raise Exception(f"Failed to export to: {path}!")
        exporter.Destroy()
        LOGGER.info(f"exported to: {path}")

    def _get_acsii_type(self):
        plg_rego     = self.manager.GetIOPluginRegistry()
        file_format  = plg_rego.GetNativeWriterFormat()
        format_count = plg_rego.GetWriterFormatCount()
        for format_index in range(format_count):
            if plg_rego.WriterIsFBX(format_index):
                desc = plg_rego.GetWriterFormatDescription(format_index)
                if "ascii" in desc:
                    file_format = format_index
                    break
        return file_format