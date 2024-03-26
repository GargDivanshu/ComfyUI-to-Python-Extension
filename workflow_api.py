import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import torchvision.transforms as T

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    NODE_CLASS_MAPPINGS,
    VAEEncode,
    ControlNetLoader,
    KSamplerAdvanced,
    VAELoader,
    ControlNetApplyAdvanced,
    VAEDecode,
    LoadImage,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    PreviewImage,
    SaveImage,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="DisneyPixarCartoontypeB.safetensors"
        )
        

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="image of a man with grey woolen coat, front pose, light brown hair",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        loadimage = LoadImage()
        loadimage_14 = loadimage.load_image(
            image="disney.png"
        )

        vaeloader = VAELoader()
        vaeloader_26 = vaeloader.load_vae(
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
        )

        vaeencode = VAEEncode()
        vaeencode_15 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_14, 0),
            vae=get_value_at_index(vaeloader_26, 0),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_19 = controlnetloader.load_controlnet(
            control_net_name="control_v11f1p_sd15_depth_fp16.safetensors"
        )

        cliptextencode_21 = cliptextencode.encode(
            text="a ((disney pixar cartoon)) highly detailed image of a man with grey woolen coat, front pose, light brown hair\nvibrant, distinct colors, with a clear, smudge-free background. Emphasize sharp details and smooth textures.",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_24 = cliptextencode.encode(
            text="text, watermark, photo, smudge, ",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        controlnetloader_29 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_canny_fp16.safetensors"
        )

        # upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        # upscalemodelloader_45 = upscalemodelloader.load_model(
        #     model_name="4xLSDIRDAT.pth"
        # )

        aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        bnk_unsampler = NODE_CLASS_MAPPINGS["BNK_Unsampler"]()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        previewimg = PreviewImage()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        midas_depthmappreprocessor = NODE_CLASS_MAPPINGS["MiDaS-DepthMapPreprocessor"]()

        for q in range(1):
            aio_preprocessor_44 = aio_preprocessor.execute(
                preprocessor="Zoe-DepthMapPreprocessor",
                resolution=512,
                image=get_value_at_index(loadimage_14, 0),
            )

            controlnetapplyadvanced_25 = controlnetapplyadvanced.apply_controlnet(
                strength=0.45,
                start_percent=0,
                end_percent=0.5,
                positive=get_value_at_index(cliptextencode_21, 0),
                negative=get_value_at_index(cliptextencode_24, 0),
                control_net=get_value_at_index(controlnetloader_19, 0),
                image=get_value_at_index(aio_preprocessor_44, 0),
            )

            aio_preprocessor_43 = aio_preprocessor.execute(
                preprocessor="LineArtPreprocessor",
                resolution=512,
                image=get_value_at_index(loadimage_14, 0),
            )

            controlnetapplyadvanced_30 = controlnetapplyadvanced.apply_controlnet(
                strength=0.45,
                start_percent=0,
                end_percent=0.5,
                positive=get_value_at_index(controlnetapplyadvanced_25, 0),
                negative=get_value_at_index(controlnetapplyadvanced_25, 1),
                control_net=get_value_at_index(controlnetloader_29, 0),
                image=get_value_at_index(aio_preprocessor_43, 0),
            )

            bnk_unsampler_13 = bnk_unsampler.unsampler(
                steps=30,
                end_at_step=0,
                cfg=1,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                normalize="disable",
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(vaeencode_15, 0),
            )

            ksampleradvanced_16 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=20,
                cfg=4.5,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                start_at_step=0,
                end_at_step=25,
                return_with_leftover_noise="disable",
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(controlnetapplyadvanced_30, 0),
                negative=get_value_at_index(controlnetapplyadvanced_30, 1),
                latent_image=get_value_at_index(bnk_unsampler_13, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_16, 0),
                vae=get_value_at_index(vaeloader_26, 0),
            )

            # imageupscalewithmodel_46 = imageupscalewithmodel.upscale(
            #     upscale_model=get_value_at_index(upscalemodelloader_45, 0),
            #     image=get_value_at_index(vaedecode_8, 0),
            # )

            midas_depthmappreprocessor_50 = midas_depthmappreprocessor.execute(
                a=6.283185307179586,
                bg_threshold=0.1,
                resolution=512,
                image=get_value_at_index(loadimage_14, 0),
            )
            saveImg = SaveImage()

            finale = saveImg.save_images(images=vaedecode_8[0], filename_prefix="OutPut")
            print("finale")
            print(finale)
            
#             output_folder = 'home/sagemaker-user-lab/Comfyui_Sagemaker/ComfyUI/output'

# # Make sure the folder exists, create it if it doesn't
#             os.makedirs(output_folder, exist_ok=True)

# # Assuming `final_tensor` is the tensor you want to save, normalized in the range [0, 1]
# # Replace this with the actual tensor you wish to save
#             final_tensor = get_value_at_index(midas_depthmappreprocessor_50, 0)  # Example placeholder

# # Convert tensor to PIL Image
#             final_image = T.ToPILImage()(final_tensor.squeeze()).convert("RGB")

# # Specify the filename
#             output_filename = "disney_output.png"

# # Combine the folder and filename to get the full path
#             full_output_path = os.path.join(output_folder, output_filename)

# # Save the image to the specified folder
#             final_image.save(full_output_path)

#             print(f"Saved final output to {full_output_path}")


if __name__ == "__main__":
    main()
