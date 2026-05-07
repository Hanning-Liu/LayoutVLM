import os
import json
import re
import textwrap
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from .sandbox import SandBoxEnv
from tqdm import tqdm
from typing import List, Dict, Literal, Optional
from utils.plot_utils import load_image, overlay_bounding_box
import base64
import collections
from utils.placement_utils import get_random_placement
from prompts.layoutvlm import base_prompt
try:
    # Rendering depends on Blender's `bpy` module. In many environments we run without Blender;
    # treat rendering as optional and fall back to no-image mode.
    from utils.blender_render import render_existing_scene
    from utils.blender_utils import reset_blender
    _BLENDER_AVAILABLE = True
except Exception:
    render_existing_scene = None
    reset_blender = None
    _BLENDER_AVAILABLE = False

from utils.blender_subprocess import is_blender_subprocess_available


def _get_render_backend() -> str:
    """Resolve at call time so callers can set LAYOUTVLM_BLENDER before LayoutVLM()."""
    if _BLENDER_AVAILABLE:
        return "inproc"
    return "subprocess" if is_blender_subprocess_available() else "none"


from collections import OrderedDict
import prompts.layoutvlm.short_prompt as short_prompt
import imageio
from PIL import Image
from utils.placement_utils import replace_z_rot_degree_to_rpy_radians
from .qwen_dashscope_client import get_dashscope_client, chat_completions_text


def extract_python_program(input_text):
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, input_text, flags=re.DOTALL)
    return matches

def extract_description_program(input_text):
    pattern = r"\*\*\*(.*?)\*\*\*"
    matches = re.findall(pattern, input_text, flags=re.DOTALL)
    return matches

def extract_json(input_text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, input_text, flags=re.DOTALL)
    return matches

class LayoutVLM:

    def __init__(self, save_dir, gpt_4o_model_name="gpt-4o", asset_source="objaverse", mode="finetuned", visual_mark_mode="new_coord", 
                 ft_original_model_id=None, ft_model_checkpoint=None, convert_z_rot_degree_to_rpy_radians=True, max_place_remaining_retry=2,
                 numerical_value_only=False):
        # initialize llm
        self.mode = mode
        self.asset_source = asset_source
        self.save_dir = save_dir
        # DashScope (Aliyun Bailian) OpenAI-compatible client (see `test_dashscope_api.py`).
        self._client = get_dashscope_client()
        self.model_name = gpt_4o_model_name
        self.model_name_mini = os.getenv("DASHSCOPE_MINI_MODEL", self.model_name)
        self.model_name_grouping = os.getenv("DASHSCOPE_GROUPING_MODEL", self.model_name)
        self.visual_mark_mode = visual_mark_mode
        self.numerical_value_only = numerical_value_only

        self.ft_original_model_id = ft_original_model_id
        self.ft_model_checkpoint = ft_model_checkpoint
        self.convert_z_rot_degree_to_rpy_radians = convert_z_rot_degree_to_rpy_radians
        self.max_place_remaining_retry = max_place_remaining_retry
        self.blender_available = _BLENDER_AVAILABLE
        self.render_backend = _get_render_backend()

        # Image-conditioned layout needs either in-process bpy or an external `blender` binary.
        if self.render_backend == "none" and self.mode != "no_image":
            print(
                "Blender rendering unavailable: no in-process bpy and no usable `blender` binary "
                "(set LAYOUTVLM_BLENDER or install `blender` on PATH). Switching to text-only (mode=no_image)."
            )
            self.mode = "no_image"
        elif self.render_backend == "inproc" and self.mode != "no_image":
            print("Blender rendering available (in-process bpy); image-conditioned mode enabled.")
        elif self.render_backend == "subprocess" and self.mode != "no_image":
            print(
                "[render-subprocess] Image-conditioned layout will spawn Blender for intermediate scene renders."
            )



    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def mark_image(self, visual_marks, input_path, output_path, image_size=(512, 512)):
        if self.visual_mark_mode == "grid":
            assert False
        elif self.visual_mark_mode == "old_coord":
            assert False
        elif self.visual_mark_mode == "new_coord":
            assert input_path is not None
            from utils.image_annotate import annotate_image_with_coordinates

            annotate_image_with_coordinates(input_path, visual_marks, output_path)
        else:
            raise NotImplementedError(f"Visual mark mode {self.visual_mark_mode} not implemented")

    def _render_scene(self, placed_assets, task, save_dir, render_kwargs):
        """Top-down (+ side) renders: in-process bpy or external Blender subprocess."""
        if _get_render_backend() == "inproc":
            if render_existing_scene is None or reset_blender is None:
                raise RuntimeError("in-process Blender render requested but bpy stack is unavailable")
            output_images, visual_marks, annotations = render_existing_scene(
                placed_assets, task, save_dir=save_dir, **render_kwargs
            )
            reset_blender()
            from utils.image_annotate import apply_annotations

            apply_annotations(annotations)
            return output_images, visual_marks
        if _get_render_backend() == "subprocess":
            from utils.blender_subprocess import render_via_blender_subprocess

            return render_via_blender_subprocess(placed_assets, task, save_dir, **render_kwargs)
        raise RuntimeError("scene render requested but no Blender backend is configured")

    def get_asset_groups(self, task, MAX_ATTEMPTS=5, save_dir=None, include_position_in_prompt=False):
        ### old version with Pydantic
        #class Assets(BaseModel):
        #    assets: List[str] = Field(description="the list of of asset names in the group")
        #    layout_criteria: str = Field(description="the layout instruction for this group of assets describing how they should be placed in the 3D scene. This instruction should mostly pertain to the assets in this group.")
        #class AssetGroups(BaseModel):
        #    group: List[Assets] = Field(description="List of grouped assets that should be placed together")
        # parser = PydanticOutputParser(pydantic_object=AssetGroups)
        # prompt = open("prompts/layoutvlm/unused/asset_grouping.txt", "r").read()
        #prompt = prompt.replace("TASK_DESCRIPTION", task["task_description"])
        # prompt.replace("OBJECT_LIST", object_list)
        #prompt = PromptTemplate(
        #    template=prompt + "\n{format_instructions}\n",
        #    input_variables=["asset_lists"],
        #    partial_variables={"format_instructions": parser.get_format_instructions()},
        #)
        #chain = prompt | self.llm_fast
        #result = chain.invoke(input={"asset_lists": object_list})
        from prompts.layoutvlm import grouping
        object_list = "[asset name] | [description] | [bounding box] \n | [position] \n" 
        for instance_id, asset in task["assets"].items():
            if include_position_in_prompt:
                object_list += f'{instance_id} | {asset["annotations"]["description"]} | {asset["assetMetadata"]["boundingBox"]} | {asset["annotations"]["position"]}\n'
            else:
                object_list += f'{instance_id} | {asset["annotations"]["description"]} | {asset["assetMetadata"]["boundingBox"]}\n'

        prompt = grouping.grouping_v1_flat_text
        prompt = prompt.replace("ASSET_LISTS", object_list)
        prompt = prompt.replace("TASK_DESCRIPTION", task["task_description"])
        prompt = prompt.replace("LAYOUT_CRITERIA", task["layout_criteria"])
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        for attempt_idx in range(MAX_ATTEMPTS):
            try:
                response_text = chat_completions_text(
                    client=self._client,
                    model=self.model_name_grouping,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.0,
                )
                if save_dir is not None:
                    with open(f"{save_dir}/grouping_{attempt_idx}.txt", "w") as f:
                        f.write(prompt + "\n\n" + response_text)
                matches = extract_json(response_text)
                result = json.loads(matches[-1])
                return result["list"]
            except Exception as e:
                print("Retrying in get_asset_groups ...", e)



    def get_initialization(self, final_prompt, encoded_image=None):
        if encoded_image is None:
            content = [{"type": "text", "text": final_prompt}]
        else:
            content = [
                {"type": "text", "text": final_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
        response_text = chat_completions_text(
            client=self._client,
            model=self.model_name_mini,
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0.0,
        )
        matches = extract_python_program(response_text)
        if matches:
            program = matches[0]
        else:
            program = ""
        return program

    def prepare_finetuned_vlm_prompt(self, final_prompt, encoded_images=[]):
        conversation = []
        user_content = [
            {
                "type": "text",
                "text": final_prompt.replace("<image>", "")
            }
        ]
        user_content.extend([{"type": "image"} for _ in range(len(encoded_images))])
        conversation.append({
            "role": "user",
            "content": user_content
        })
        image_list = [Image.open(image_path).convert("RGB") for image_path in encoded_images]
        return conversation, image_list

    def get_constraint_program(self, final_prompt, current_scene_image_path_dict, current_group_asset_img_path_dict, program_save_path=None):
        image_paths = []
        if self.mode != "no_image":
            top_down_scene_image_path = current_scene_image_path_dict["top_down_rendering"]
            side_scene_image_path = current_scene_image_path_dict["side_rendering_45_3"]
            asset_images = [current_group_asset_img_path_dict[asset_name] for asset_name in current_group_asset_img_path_dict]
            image_paths = [top_down_scene_image_path, side_scene_image_path] + asset_images
        

        messages = [
            {
                "role": "system",
                "content": "You are a coding agent. PLEASE DO NOT REPEAT ANY OF THE CODE CODE GIVEN. DO NOT RE-INITIALIZE ANY OF THE GIVEN VARIABLES.",
            }
        ]
        content = [{"type": "text", "text": final_prompt}]

        encoded_images = [self.encode_image(image_path) for image_path in image_paths]
        for encoded_image in encoded_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                }
            )
        messages.append({"role": "user", "content": content})
        response_text = chat_completions_text(
            client=self._client,
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )
        if self.convert_z_rot_degree_to_rpy_radians:
            response_text = replace_z_rot_degree_to_rpy_radians(response_text)

        with open(program_save_path, "w") as f:
            f.write(response_text)
        matches = extract_python_program(response_text)
        if matches:
            constraint_program = matches[0]
        else:
            constraint_program = response_text

        ### remove re-initialized variables
        matches = list(re.finditer(r"\w+ = Assets\(", constraint_program))
        if matches:
            last_match = matches[-1]
            # Extract the code after the last mat    
            end_of_line = constraint_program.find("\n", last_match.end())
            # Extract the code after the last match's line
            constraint_program = constraint_program[end_of_line + 1:].strip()
            # matches = list(re.finditer(r"\w+ = Assets\(", constraint_program))
        return constraint_program

    @staticmethod
    def get_task_program(grouped_assets, task, verify_asset_var_name_to_count=None):
        """
        Args:
            grouped_assets: list of grouped assets
            task: input json of the scene and the assets
            verify_asset_var_name_to_count: used to verify whether count is correct
        """
        program = "# Walls that define the boundary of the scene\n"
        floor_vertices = task['boundary']['floor_vertices']
        num_walls = len(floor_vertices)
        program += "walls = [\n"
        for wall_idx in range(len(task["boundary"]["floor_vertices"])):
            size_str1 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[wall_idx][0],floor_vertices[wall_idx][1], floor_vertices[wall_idx][2]
            )
            size_str2 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[(wall_idx+1)%num_walls][0],floor_vertices[(wall_idx+1)%num_walls][1], floor_vertices[(wall_idx+1)%num_walls][2]
            )
            if wall_idx == len(task["boundary"]["floor_vertices"]) - 1:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2})\n]\n"
            else:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2}),\n"

        program += "\n# Existing assets placed in the scene:\n"
        uid2asset = dict()
        for instance_uid, asset in task['assets'].items():
            if instance_uid in grouped_assets:
                continue
            #printasset)
            asset_uid = asset["asset_var_name"]
            if asset_uid not in uid2asset.keys():
                uid2asset[asset_uid] = {
                    "asset": asset,
                    "count": 1
                }
            else:
                uid2asset[asset_uid]["count"] += 1

        for asset_uid, value in uid2asset.items():
            asset = value["asset"]
            size_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
                asset['assetMetadata']['boundingBox']['x'],
                asset['assetMetadata']['boundingBox']['y'],
                asset['assetMetadata']['boundingBox']['z']
            )
            if verify_asset_var_name_to_count is not None:
                assert value['count'] == verify_asset_var_name_to_count[asset['asset_var_name']], (
                    "value['count'] ({}) != verify_asset_var_name_to_count[asset['asset_var_name']] ({}) for {}".format(
                        value['count'], verify_asset_var_name_to_count[asset['asset_var_name']], asset['asset_var_name']))
            program += (f"{asset['asset_var_name']} = Assets("
                f"description=\"{asset['description']}\", "
                f"size={size_str}, "
                f"placements=[AssetInstance() for _ in range({value['count']})])\n"
            )

        program += '\n# New assets to be placed\n'
        ### FORM for loops
        uid2asset = dict()
        for instance_uid, asset in task['assets'].items():
            if instance_uid not in grouped_assets:
                continue
            asset_uid = asset["asset_var_name"]
            if asset_uid not in uid2asset.keys():
                uid2asset[asset_uid] = {
                    "asset": asset,
                    "count": 1
                }
            else:
                uid2asset[asset_uid]["count"] += 1

        for asset_uid, value in uid2asset.items():
            asset = value["asset"]
            size_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
                asset['assetMetadata']['boundingBox']['x'],
                asset['assetMetadata']['boundingBox']['y'],
                asset['assetMetadata']['boundingBox']['z']
            )
            if verify_asset_var_name_to_count is not None:
                assert value['count'] == verify_asset_var_name_to_count[asset['asset_var_name']], (
                    "value['count'] ({}) != verify_asset_var_name_to_count[asset['asset_var_name']] ({}) for {}".format(
                        value['count'], verify_asset_var_name_to_count[asset['asset_var_name']], asset['asset_var_name']))
            program += (f"{asset['asset_var_name']} = Assets("
                f"description=\"{asset['description']}\", "
                f"size={size_str}, "
                f"placements=[AssetInstance() for _ in range({value['count']})])\n"
            )
        return program

    def _solve_single_group(self, task, layout_criteria, placed_assets, group_assets, _save_dir, 
                            include_image=True, MAX_ATTEMPTS=5, only_initialize=False):
        #############################################################################
        ### prepare scene images
        #############################################################################
        current_scene_image_path_dict = {}
        if include_image:
            if _get_render_backend() == "none":
                print(
                    "Note: include_image=True but no Blender backend is available; "
                    "skipping scene image rendering for this group."
                )
                include_image = False
            elif _get_render_backend() == "inproc" and (
                render_existing_scene is None or reset_blender is None
            ):
                print(
                    "Note: include_image=True but in-process Blender utilities failed to load; "
                    "skipping scene image rendering for this group."
                )
                include_image = False
            else:
                common = dict(
                    high_res=True,
                    render_top_down=True,
                    apply_3dfront_texture=True,
                    combine_obj_components=True,
                    fov_multiplier=1.3,
                )
                if self.mode == "no_visual_coordinate":
                    render_kwargs = {
                        **common,
                        "add_coordinate_mark": False,
                        "annotate_object": True,
                        "annotate_wall": True,
                        "add_object_bbox": False,
                    }
                elif self.mode == "no_visual_assetname":
                    render_kwargs = {
                        **common,
                        "add_coordinate_mark": True,
                        "annotate_object": False,
                        "annotate_wall": False,
                        "add_object_bbox": False,
                    }
                elif self.mode == "no_visual_mark":
                    render_kwargs = {
                        **common,
                        "add_coordinate_mark": False,
                        "annotate_object": False,
                        "annotate_wall": False,
                        "add_object_bbox": False,
                    }
                else:
                    render_kwargs = dict(common)

                output_images, _visual_marks = self._render_scene(
                    placed_assets, task, _save_dir, render_kwargs
                )
                ### render the incomplete scene with the visual marks
                for _image in output_images:
                    current_scene_image_path_dict[os.path.basename(_image).split('.')[0]] = _image

        #############################################################################
        ### prepare asset images (omitted)
        #############################################################################
        current_group_asset_img_path_dict = OrderedDict()

        #############################################################################
        ### form the prompt get the list of asset names in the group
        #############################################################################
        _task = task.copy()
        _task["assets"] = {k: v for k, v in task["assets"].items() if (k in placed_assets.keys() or k in group_assets)}
        task_program_for_prompt = self.get_task_program(group_assets, _task)

        asset_placed_list = [asset_name.replace('-', '[') + ']' if '-' in asset_name else asset_name for asset_name in task['assets'].keys() if asset_name not in group_assets]
        asset_be_placed_list = [asset_name.replace('-', '[') + ']' if '-' in asset_name else asset_name for asset_name in group_assets]
        #pattern = re.compile(r"(\b\w+)\[(\d+)\]")
        def split_asset_string(asset):
            # Regex to match the pattern "{asset_type}[{index}]"
            match = re.match(r"(\w+)\[(\d+)\]", asset)
            if match:
                asset_type, index = match.groups()
                return asset_type, int(index)
            else:
                return None, None
        index_map = {}
        for asset in asset_be_placed_list:
            asset_type, index = split_asset_string(asset)
            if asset_type is None:
                print("Asset string format is incorrect: ", asset)
                continue
            if asset_type in index_map:
                index_map[asset_type] = min(index_map[asset_type], index)
            else:
                index_map[asset_type] = index
        # Step 2: Adjust indices to start from 0 for each asset type
        normalized_assets = []
        for asset in asset_be_placed_list:
            asset_type, index = split_asset_string(asset)
            if asset_type is None:
                print("Asset string format is incorrect: ", asset)
                continue
            # Subtract the minimum index found for this asset type to normalize
            new_index = int(index) - index_map[asset_type]
            normalized_assets.append(f"{asset_type}[{new_index}]")
        replacement_map = dict(zip(normalized_assets, asset_be_placed_list))
        # note: use the newest prompt
        # final_prompt = base_prompt.get_layout_prompt(task_program_for_prompt, layout_criteria)
        final_prompt = short_prompt.get_layout_prompt(task_program_for_prompt, {"layout_criteria": layout_criteria}, self.numerical_value_only)

        os.makedirs(_save_dir, exist_ok=True)
        with open(f"{_save_dir}/prompt.txt", "w") as f:
            f.write(final_prompt)
            #f.write(f"Asset_index: {', '.join(asset_be_placed_list)}")


        for attempt_idx in range(MAX_ATTEMPTS):
            # try:
            if True:
                # clear constraints
                self.sandbox.execute_code("solver.constraints = []\n")
                save_path = f"{_save_dir}/llm_output_program_{attempt_idx}.py"
                constraint_program = self.get_constraint_program(final_prompt,
                                                                 current_scene_image_path_dict,
                                                                 current_group_asset_img_path_dict,
                                                                 program_save_path=save_path)
                
                # print("BEFORE")
                # print(constraint_program)
                # Perform replacement
                for old_asset, new_asset in replacement_map.items():
                    constraint_program = constraint_program.replace(old_asset, new_asset)
                # Some models return top-level code with leading indentation, which breaks `exec`.
                # Normalize indentation without changing the relative indentation inside blocks.
                constraint_program = textwrap.dedent(constraint_program).lstrip()

                # Normalize instance references. The sandbox defines instances as `asset_var[idx]`,
                # but some models emit `asset_var_idx` (e.g. `tv_console_0`).
                try:
                    asset_var_names = {a["asset_var_name"] for a in task.get("assets", {}).values() if "asset_var_name" in a}
                    if asset_var_names:
                        def _fix_instance_ref(m):
                            base = m.group(1)
                            idx = m.group(2)
                            if base in asset_var_names:
                                return f"{base}[{idx}]"
                            return m.group(0)
                        constraint_program = re.sub(r"\b([A-Za-z_]\w*)_(\d+)\b", _fix_instance_ref, constraint_program)
                except Exception:
                    pass
                # print("AFTER")
                # print(constraint_program)
                #import pdb; pdb.set_trace()
                if constraint_program == "":
                    print("Constraint program is empty")
                    continue
                # find the last line of code with the pattern * = Assets(...)
                # get the constraint program only after the last line
                self.sandbox.sanity_check(group_assets, constraint_program)
                placed_assets = self.sandbox.solve(
                    placed_assets, group_assets, constraint_program, save_dir=_save_dir, only_initialize=only_initialize
                )
                break
            # except Exception as e:
            #     print("Retrying ...", e)

        return placed_assets

    def solve(self, original_task, MAX_ATTEMPTS=3):
        """
        task is the input json of the scene and the assets 
        """
        task = original_task.copy()
        #### initialize the sandbox and initialize all the variables
        self.sandbox = SandBoxEnv(task, mode=self.mode, save_dir=self.save_dir)
        task_program = self.get_task_program(list(task["assets"].keys()), task)
        self.sandbox.execute_code(base_prompt.CODE_FOR_SANDBOX + "\n" + task_program)
        self.sandbox.assign_instance_ids()
        self.sandbox.initialize_variables()
        self.sandbox.export_code()

        include_image = self.mode != "no_image"
        ### get asset groupings
        if self.mode == "one_shot":
            placed_assets = dict()
            unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
            num_groups = 0
            while len(unplaced_assets) and num_groups < 20:
                print(f"Placing unplaced assets -- group {num_groups}")
                _save_dir = os.path.join(self.save_dir, f"group_{num_groups}")
                placed_assets = self._solve_single_group(
                    task, task["layout_criteria"],
                    placed_assets, unplaced_assets, _save_dir, include_image=include_image, MAX_ATTEMPTS=MAX_ATTEMPTS
                )
                unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
                num_groups += 1

        else:
            group_list = self.get_asset_groups(task, save_dir=self.save_dir, MAX_ATTEMPTS=MAX_ATTEMPTS)
            with open(self.save_dir + "/grouping.json", "w") as f:
                json.dump(group_list, f, indent=4)

            placed_assets = self.sandbox.export_layout(incomplete_scene=True, use_degree=True)
            for group_idx, group in enumerate(group_list):
                _save_dir = os.path.join(self.save_dir, f"group_{group_idx}")
                os.makedirs(_save_dir, exist_ok=True)
                ### optionally: use another prompt to select the relevant context in the scene
                ### For now, we are using the top-down image plus the whole list of assets of the scene as the context

                layout_criteria = f"{task['layout_criteria']}. More specifically, Organize the {group['name']} of the room in the following way:\n"
                _key_relations_between_assets = "\n".join(group['key_relations_between_assets'])
                layout_criteria += f"{_key_relations_between_assets}\n"
                placed_assets = self._solve_single_group(
                    task, layout_criteria, placed_assets, group['assets'],
                    _save_dir, include_image=include_image, MAX_ATTEMPTS=MAX_ATTEMPTS,
                )
            # Find assets that are not placed yet
            num_groups = len(group_list)

        unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
        num_place_remaining_retry = 0
        # while not empty
        while len(unplaced_assets) and num_groups < 20 and num_place_remaining_retry < self.max_place_remaining_retry:
            print(f"Placing unplaced assets -- group {num_groups}")
            _save_dir = os.path.join(self.save_dir, f"group_{num_groups}")
            placed_assets = self._solve_single_group(
                task, task["layout_criteria"],
                placed_assets, unplaced_assets, _save_dir, include_image=include_image, MAX_ATTEMPTS=MAX_ATTEMPTS
            )
            unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
            num_groups += 1
            num_place_remaining_retry += 1

        if len(unplaced_assets) == 0:
            print("All assets have already been placed.")

        results = self.sandbox.export_layout(use_degree=True)
        ### save into one final gif
        if self.mode not in ["no_constraint", "finetuned"]:
            all_frames = []
            gif_files = []
            for group_idx in range(num_groups):
                if os.path.exists(f"{self.sandbox.save_dir}/group_{group_idx}/out.gif"):
                    gif_files.append(f"{self.sandbox.save_dir}/group_{group_idx}/out.gif")
            for gif_file in gif_files:
                gif = imageio.mimread(gif_file)  # Read all frames from the GIF
                all_frames.extend(gif)
            if len(all_frames) > 0:
                imageio.mimsave(f"{self.save_dir}/final.gif", all_frames)

        return results

    def get_simple_program(grouped_assets, task):
        """
        Args:
            grouped_assets: list of grouped assets
            task: input json of the scene and the assets
        """
        program = "# Walls that define the boundary of the scene\n"
        floor_vertices = task['boundary']['floor_vertices']
        num_walls = len(floor_vertices)
        program += "walls = [\n"
        for wall_idx in range(len(task["boundary"]["floor_vertices"])):
            size_str1 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[wall_idx][0],floor_vertices[wall_idx][1], floor_vertices[wall_idx][2]
            )
            size_str2 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[(wall_idx+1)%num_walls][0],floor_vertices[(wall_idx+1)%num_walls][1], floor_vertices[(wall_idx+1)%num_walls][2]
            )
            if wall_idx == len(task["boundary"]["floor_vertices"]) - 1:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2})\n]\n"
            else:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2}),\n"

        ### FORM for loops
        new_asset_list  = []
        uid2asset = dict()
        for instance_uid, asset in task['assets'].items():
            print(instance_uid)
            new_asset_list.append(instance_uid)
            asset_uid = asset["asset_var_name"]
            if asset_uid not in uid2asset.keys():
                uid2asset[asset_uid] = {
                    "asset": asset,
                    "count": 1
                }
            else:
                uid2asset[asset_uid]["count"] += 1

        program += f"\n# New assets to be placed: [{', '.join(new_asset_list)}]\n"
        for asset_uid, value in uid2asset.items():
            asset = value["asset"]
            size_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
                asset['assetMetadata']['boundingBox']['x'],
                asset['assetMetadata']['boundingBox']['y'],
                asset['assetMetadata']['boundingBox']['z']
            )
            program += (f"{asset['asset_var_name']} = Assets("
                f"description=\"{asset['description']}\", "
                f"size={size_str}, "
                f"placements=[AssetInstance() for _ in range({value['count']})])\n"
            )
        
        return program
    

    def filter_constraint(self, image_path, final_prompt, save_path = None):
        # print(image_path)
        # print(final_prompt)
        with open(image_path, "rb") as image_file:
            encoded_image =  base64.b64encode(image_file.read()).decode('utf-8')
        messages = [{"role": "system", "content": "You are a coding agent."}]
        content = [{"type": "text", "text": final_prompt}]
        
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )
        messages.append({"role": "user", "content": content})

        for attempt_idx in range(3):
            try:
                response_text = chat_completions_text(
                    client=self._client,
                    model=self.model_name_mini,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.0,
                )
                
                # print(response_text)
                # with open(save_path, "w") as f:
                #     f.write(response_text)
                constraint_program = extract_python_program(response_text)
                if constraint_program:
                    constraint_program = constraint_program[0]
                else:
                    constraint_program = ""

                task_description = extract_description_program(response_text)
                if task_description:
                    task_description = task_description[0]
                else:
                    task_description = ""
                #print(constraint_program)  
                
                return constraint_program, task_description
                
            except Exception as e:
                print("Retrying in filter_constraint ...", e)
